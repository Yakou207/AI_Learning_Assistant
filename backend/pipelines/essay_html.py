from __future__ import annotations

import json
from typing import Any, Callable

from backend.artifacts.filesystem import ArtifactRun
from backend.pipelines.authorship import add_author_credit
from backend.pipelines.common import (
    PipelineError,
    PipelineResult,
    extract_fenced_or_raw,
    format_citations,
    format_log,
)
from backend.pipelines.html_to_pdf import (
    A4_PAGE,
    ConvertError,
    HtmlToPdfConverter,
)
from backend.providers.base import (
    ModelProviderError,
    TextGenerationProvider,
    TextGenerationRequest,
)
from backend.timing import RunTimingRecorder, measure_stage


def run_essay_html_pipeline(
    *,
    artifact_run: ArtifactRun,
    model_profile: dict[str, Any],
    model_provider: TextGenerationProvider,
    pdf_converter: HtmlToPdfConverter,
    task_text: str,
    context_bundle: str,
    output_preference: str | None,
    options: dict[str, Any] | None,
    search: dict[str, Any],
    max_output_tokens: int,
    emit_event: Callable[[str, str], None] | None = None,
    timing: RunTimingRecorder | None = None,
) -> PipelineResult:
    log_lines = [
        "Pipeline: essay_html",
        f"Output preference: {output_preference or 'pdf'}",
        "Stage: generate_source",
    ]
    system_prompt, user_prompt = build_essay_html_prompt(
        task_text=task_text,
        context_bundle=context_bundle,
        options=options or {},
        search=search,
    )
    if emit_event:
        emit_event("generate_source", "Generating essay HTML source")

    try:
        with measure_stage(timing, "provider_generation"):
            raw_output = model_provider.generate_text(
                TextGenerationRequest(
                    profile=model_profile,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=max_output_tokens,
                    temperature=0.2,
                )
            )
    except ModelProviderError as exc:
        raise PipelineError(
            exc.code,
            exc.message,
            stage="generate_source",
            log_lines=log_lines,
        ) from exc
    except Exception as exc:
        raise PipelineError(
            "provider_unavailable",
            "The model provider request failed.",
            stage="generate_source",
            log_lines=log_lines,
        ) from exc

    source = extract_fenced_or_raw(raw_output, accepted_languages={"html", "htm"})
    source = add_author_credit(source.strip()) + "\n"
    html_path = artifact_run.write_output(
        "main.html",
        source,
        kind="source",
        media_type="text/html",
    )
    log_lines.extend(["Output: output/main.html", "Stage: convert_pdf"])
    if emit_event:
        emit_event("convert_pdf", "Converting essay HTML to PDF")

    pdf_path = artifact_run.run_dir / "output" / "main.pdf"
    try:
        with measure_stage(timing, "convert_pdf"):
            convert_result = pdf_converter.convert(
                html_path=html_path,
                pdf_path=pdf_path,
                page_config=A4_PAGE,
            )
    except ConvertError as exc:
        artifact_run.write_log("convert.log", exc.log_text)
        raise PipelineError(
            "convert_failed",
            exc.message,
            stage="convert_pdf",
            log_lines=log_lines,
        ) from exc
    except Exception as exc:
        artifact_run.write_log(
            "convert.log",
            "HTML-to-PDF converter raised an unexpected error.\n",
        )
        raise PipelineError(
            "convert_failed",
            "HTML-to-PDF conversion failed.",
            stage="convert_pdf",
            log_lines=log_lines,
        ) from exc

    artifact_run.write_log("convert.log", convert_result.log_text)
    artifact_run.write_output(
        "main.pdf",
        convert_result.pdf_path.read_bytes(),
        kind="pdf",
        media_type="application/pdf",
    )
    log_lines.extend(["Convert: pdf_ok", "Output: output/main.pdf"])
    return PipelineResult(log_text=format_log(log_lines))


def build_essay_html_prompt(
    *,
    task_text: str,
    context_bundle: str,
    options: dict[str, Any],
    search: dict[str, Any],
) -> tuple[str, str]:
    system_prompt = (
        "You are an expert teaching assistant writing complete academic essay and "
        "report deliverables as self-contained HTML documents. You are also an "
        "experienced cloud engineer who can explain deployments clearly to a "
        "university marker. Return only the requested HTML source, with no "
        "surrounding explanation."
    )
    report_instruction = (
        "Decide whether the task asks for an essay or for a technical report. "
        "When it asks for a technical, lab, deployment, or assignment report, "
        "follow these rules.\n"
        "1. Structure: a title block (title, course, author details only if "
        "provided, date); a short executive summary; an overall architecture "
        "section; one section per assignment task in the order the task lists "
        "them; a security and cost analysis; limitations and improvements; a "
        "conclusion; and an appendix only if evidence would otherwise crowd the "
        "main text. Number sections and figures (Figure 1, Figure 2, ...) and "
        "tables (Table 1, ...), and refer to them in the text.\n"
        "2. Map every marked requirement in the task to a section, and end with "
        "a requirement checklist table (requirement, where it is addressed, "
        "evidence) so the marker can find each item quickly.\n"
        "3. For each task, explain what was done, why it was done that way, and "
        "how it was verified. Show the essential commands or configuration in "
        "short <pre><code> blocks, followed by one or two sentences that explain "
        "what they do. Do not paste long logs; quote only the lines that prove "
        "the result.\n"
        "4. Explain how the deployed system works end to end: the request path "
        "from the user's browser through DNS, any CDN or reverse proxy, TLS, "
        "access control, the container, the application, storage, and the model "
        "API. Draw it as a simple inline SVG diagram with labelled boxes and "
        "arrows that fits the page width.\n"
        "5. Use only evidence supplied in the task or prepared context: commands, "
        "outputs, check-file contents, IP addresses, domains, versions, and "
        "timestamps. Never invent results, numbers, or outputs. If a piece of "
        "evidence is missing, say so plainly instead of guessing.\n"
        "6. Never reveal secrets. Replace API keys, passwords, tokens, private "
        "keys, and password hashes with [REDACTED], even if they appear in the "
        "context.\n"
        "7. For each screenshot the task asks for, insert a figure placeholder "
        "exactly like <figure class=\"evidence\" data-evidence=\"ID\"><div "
        "class=\"evidence-slot\">Screenshot: ID</div><figcaption>Figure N. "
        "caption</figcaption></figure>, where ID is the screenshot identifier "
        "given in the task (or a short kebab-case name if none is given) and the "
        "caption states what the screenshot proves. Style .evidence-slot as a "
        "bordered box about 90mm tall so the screenshot can be inserted later.\n"
        "8. Security and cost analysis must be specific to the deployment: "
        "exposed ports, firewall or security-group rules, TLS, access control, "
        "secret storage, data persistence and backup, and the main recurring "
        "costs with ways to reduce them.\n"
        "9. Improvements must be concrete and prioritised (for example stronger "
        "authentication, monitoring, backups, CI/CD, scaling), each with its "
        "benefit and trade-off.\n"
        "10. Write in clear, formal, concise English for a non-specialist "
        "reader. Define each technical term the first time it is used. Respect "
        "any page limit in the task; a 20-page limit means roughly 5,000 words "
        "plus figures at most. Avoid page breaks inside figures, tables, and "
        "code blocks."
    )
    output_instruction = (
        "Return one complete self-contained HTML document. It must start with "
        "<!doctype html> and include <html>, <head>, and <body>. Include all CSS "
        "inside a <style> element: use a readable serif academic print layout, "
        "clear heading hierarchy, careful paragraph spacing, and CSS "
        "@page { size: A4; margin: 20mm; }. Do not use external stylesheet links, "
        "remote images, or remote HTTP(S) assets. If mathematical notation is "
        "needed, include inline KaTeX-compatible markup and any needed minimal "
        "KaTeX styling inline in the document; do not fetch KaTeX from a CDN at "
        "render time. Use inline SVG or CSS for diagrams. Cite any provided web "
        "sources in a References section using ordinary HTML. Do not wrap the "
        "answer in Markdown fences."
    )
    user_prompt = "\n\n".join(
        [
            "[Assignment Task]\n" + task_text.strip(),
            "[Prepared Context]\n" + context_bundle.strip(),
            "[Search Citations]\n" + format_citations(search),
            "[Options]\n" + json.dumps(options, sort_keys=True),
            "[Report Guidance]\n" + report_instruction,
            "[Output Contract]\n" + output_instruction,
        ]
    )
    return system_prompt, user_prompt
