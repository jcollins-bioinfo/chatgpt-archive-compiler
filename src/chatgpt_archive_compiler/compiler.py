"""Local Archive IR analysis, document construction, and HTML/PDF compilation."""

from __future__ import annotations

import hashlib
import html
import importlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, TextIO, cast
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from pydantic import BaseModel, ConfigDict, Field

from chatgpt_archive_compiler.exceptions import ArchiveCompilationError
from chatgpt_archive_compiler.models import (
    Archive,
    ContentBlock,
    ContentBlockType,
    Conversation,
    Message,
    Role,
)
from chatgpt_archive_compiler.version import __version__


class VolumeMode(StrEnum):
    """How selected conversations are partitioned into output volumes."""

    SINGLE = "single"
    MONTH = "month"
    YEAR = "year"


class RedactionRule(BaseModel):
    """One explicit regular-expression replacement applied to rendered private text."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    pattern: str = Field(min_length=1)
    replacement: str = "[REDACTED]"


class CompileOptions(BaseModel):
    """Configuration for deterministic local archive compilation.

    Reasoning traces and summaries, non-user-facing roles, alternate branches, and remote assets
    are excluded by default. Source content remains preserved in Archive IR.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    title: str = "ChatGPT Conversation Archive"
    subtitle: str | None = None
    author: str | None = None
    volume_mode: VolumeMode = VolumeMode.MONTH
    render_pdf: bool = False
    include_reasoning_summaries: bool = False
    include_system_messages: bool = False
    include_tool_messages: bool = False
    max_conversations: int | None = Field(default=None, gt=0)
    redaction_rules: tuple[RedactionRule, ...] = ()


class CorpusAnalysis(BaseModel):
    """Privacy-safe structural counts derived from one Archive IR."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    conversation_count: int = Field(ge=0)
    graph_node_count: int = Field(ge=0)
    graph_message_count: int = Field(ge=0)
    current_path_message_count: int = Field(ge=0)
    alternate_path_message_count: int = Field(ge=0)
    messages_by_role: dict[str, int]
    blocks_by_type: dict[str, int]
    conversations_by_year: dict[str, int]
    warnings_by_code: dict[str, int]
    earliest_conversation_at: datetime | None = None
    latest_conversation_at: datetime | None = None


class CompiledVolume(BaseModel):
    """Paths and non-content integrity metadata for one compiled volume."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    label: str
    conversation_count: int = Field(ge=0)
    message_count: int = Field(ge=0)
    html_path: Path
    html_sha256: str
    pdf_path: Path | None = None
    pdf_sha256: str | None = None


class CompilationResult(BaseModel):
    """Complete result of compiling an Archive IR into local private artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    output_directory: Path
    analysis_path: Path
    index_path: Path
    manifest_path: Path
    selected_conversation_count: int = Field(ge=0)
    volumes: tuple[CompiledVolume, ...]


class _PDFDocument(Protocol):
    """Structural type for the optional WeasyPrint document surface."""

    def write_pdf(self, target: str) -> object:
        """Write the rendered PDF to ``target``."""


class _HTMLFactory(Protocol):
    """Structural type for restricted local WeasyPrint construction."""

    def __call__(
        self,
        *,
        filename: str,
        base_url: str,
        url_fetcher: Callable[..., object],
    ) -> _PDFDocument:
        """Construct one document with a caller-supplied subsidiary-resource policy."""


_MARKDOWN = MarkdownIt("commonmark", {"html": False, "breaks": True})
_REASONING_TYPES = {
    ContentBlockType.THINKING_TRACE,
    ContentBlockType.REASONING_SUMMARY,
}


@contextmanager
def _atomic_text_writer(destination: Path) -> Iterator[TextIO]:
    """Yield a UTF-8 writer that atomically replaces its destination on success."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            yield cast(TextIO, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    except OSError as exc:
        raise ArchiveCompilationError(
            f"Could not write compilation artifact: {destination}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_json(value: BaseModel | dict[str, object], destination: Path) -> Path:
    """Write one canonical JSON artifact atomically."""

    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    with _atomic_text_writer(destination) as handle:
        json.dump(data, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return destination


def _sha256(path: Path) -> str:
    """Hash one generated artifact without retaining it in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_archive(archive: Archive) -> CorpusAnalysis:
    """Return structural corpus counts without titles, message text, IDs, or source paths."""

    messages_by_role: Counter[str] = Counter()
    blocks_by_type: Counter[str] = Counter()
    conversations_by_year: Counter[str] = Counter()
    graph_node_count = 0
    graph_message_count = 0
    current_path_message_count = 0
    dates: list[datetime] = []

    for conversation in archive.conversations:
        conversation_date = conversation.created_at or conversation.updated_at
        conversations_by_year[
            str(conversation_date.year) if conversation_date is not None else "undated"
        ] += 1
        if conversation_date is not None:
            dates.append(conversation_date)
        graph_node_count += len(conversation.nodes)
        for node in conversation.nodes:
            if node.message is None:
                continue
            graph_message_count += 1
            messages_by_role[node.message.role.value] += 1
            blocks_by_type.update(block.type.value for block in node.message.content)
        current_path_message_count += len(conversation.current_path_messages)

    return CorpusAnalysis(
        conversation_count=len(archive.conversations),
        graph_node_count=graph_node_count,
        graph_message_count=graph_message_count,
        current_path_message_count=current_path_message_count,
        alternate_path_message_count=max(0, graph_message_count - current_path_message_count),
        messages_by_role=dict(sorted(messages_by_role.items())),
        blocks_by_type=dict(sorted(blocks_by_type.items())),
        conversations_by_year=dict(sorted(conversations_by_year.items())),
        warnings_by_code=dict(sorted(Counter(item.code for item in archive.all_warnings).items())),
        earliest_conversation_at=min(dates) if dates else None,
        latest_conversation_at=max(dates) if dates else None,
    )


def _conversation_sort_key(conversation: Conversation) -> tuple[float, str, str]:
    """Return a stable chronological key without depending on private display text."""

    date = conversation.created_at or conversation.updated_at
    timestamp = date.timestamp() if date is not None else float("inf")
    return timestamp, str(conversation.source_path), conversation.conversation_id or ""


def _included_roles(options: CompileOptions) -> set[Role]:
    """Resolve the user-facing role policy from compile options."""

    roles = {Role.USER, Role.ASSISTANT}
    if options.include_system_messages:
        roles.update({Role.SYSTEM, Role.DEVELOPER})
    if options.include_tool_messages:
        roles.add(Role.TOOL)
    return roles


def _message_is_visible(message: Message, options: CompileOptions) -> bool:
    """Return whether a current-path message belongs in the rendered document."""

    if message.role not in _included_roles(options):
        return False
    visible_blocks = [
        block
        for block in message.content
        if options.include_reasoning_summaries or block.type not in _REASONING_TYPES
    ]
    return bool(visible_blocks)


def _visible_messages(conversation: Conversation, options: CompileOptions) -> list[Message]:
    """Select renderable messages only from the declared current conversation path."""

    return [
        message
        for message in conversation.current_path_messages
        if _message_is_visible(message, options)
    ]


def _group_conversations(
    conversations: Sequence[Conversation], options: CompileOptions
) -> list[tuple[str, list[Conversation]]]:
    """Partition selected conversations into deterministic single, monthly, or annual volumes."""

    if options.volume_mode is VolumeMode.SINGLE:
        return [("complete", list(conversations))]

    groups: defaultdict[str, list[Conversation]] = defaultdict(list)
    for conversation in conversations:
        date = conversation.created_at or conversation.updated_at
        if date is None:
            label = "undated"
        elif options.volume_mode is VolumeMode.MONTH:
            label = f"{date.year:04d}-{date.month:02d}"
        else:
            label = str(date.year)
        groups[label].append(conversation)
    labels = sorted(label for label in groups if label != "undated")
    if "undated" in groups:
        labels.append("undated")
    return [(label, groups[label]) for label in labels]


def _compile_redactors(options: CompileOptions) -> tuple[tuple[re.Pattern[str], str], ...]:
    """Validate explicit redaction expressions before writing private artifacts."""

    compiled: list[tuple[re.Pattern[str], str]] = []
    for rule in options.redaction_rules:
        try:
            compiled.append((re.compile(rule.pattern), rule.replacement))
        except re.error as exc:
            raise ArchiveCompilationError("A configured redaction pattern is invalid.") from exc
    return tuple(compiled)


def _redact(text: str, redactors: Sequence[tuple[re.Pattern[str], str]]) -> str:
    """Apply explicit redactions and normalize Unicode separators for HTML layout engines."""

    redacted = text
    for pattern, replacement in redactors:
        redacted = pattern.sub(replacement, redacted)
    return redacted.replace("\u2028", "\n").replace("\u2029", "\n\n")


def _safe_markdown(text: str, redactors: Sequence[tuple[re.Pattern[str], str]]) -> str:
    """Render Markdown with raw HTML and remote/image asset loading disabled."""

    fragment = _MARKDOWN.render(_redact(text, redactors))
    soup = BeautifulSoup(fragment, "html.parser")
    for image in soup.find_all("img"):
        alt = image.get("alt")
        image.replace_with(f"[Image omitted: {alt}]" if alt else "[Image omitted]")
    for link in soup.find_all("a"):
        href = link.get("href")
        if not isinstance(href, str):
            link.unwrap()
            continue
        parsed = urlsplit(href)
        if href.startswith("#") or parsed.scheme.casefold() in {"http", "https", "mailto"}:
            link["rel"] = "noreferrer"
        else:
            link.unwrap()
    return str(soup)


def _render_block(
    block: ContentBlock,
    *,
    options: CompileOptions,
    redactors: Sequence[tuple[re.Pattern[str], str]],
) -> str:
    """Render one normalized content block without reading remote or referenced assets."""

    if block.type in _REASONING_TYPES:
        if not options.include_reasoning_summaries:
            return ""
        label = (
            "Thinking trace"
            if block.type is ContentBlockType.THINKING_TRACE
            else "Reasoning summary"
        )
        body = _safe_markdown(block.text or "[Preserved without displayable text]", redactors)
        return f'<aside class="reasoning"><strong>{label}</strong>{body}</aside>'
    if block.type in {ContentBlockType.MARKDOWN, ContentBlockType.TEXT}:
        return _safe_markdown(block.text or "", redactors)
    if block.type is ContentBlockType.CODE:
        language = html.escape(block.language or "text")
        text = html.escape(_redact(block.text or "", redactors))
        return f'<div class="code-label">{language}</div><pre><code>{text}</code></pre>'
    if block.type in {ContentBlockType.TOOL_CALL, ContentBlockType.TOOL_RESULT}:
        text = html.escape(_redact(block.text or "", redactors))
        return f'<pre class="tool"><code>{text}</code></pre>'
    if block.type in {
        ContentBlockType.FILE_REFERENCE,
        ContentBlockType.IMAGE_REFERENCE,
        ContentBlockType.AUDIO_REFERENCE,
    }:
        label = block.type.value.replace("_", " ").title()
        reference = html.escape(_redact(block.reference or "unavailable", redactors))
        return f'<p class="reference"><strong>{label}:</strong> <code>{reference}</code></p>'
    return '<p class="unsupported">Unsupported content is preserved in Archive IR.</p>'


def _display_date(conversation: Conversation) -> str:
    """Return a stable human-readable conversation date."""

    date = conversation.created_at or conversation.updated_at
    if date is None:
        return "Undated"
    utc_date = date.astimezone(UTC)
    return f"{utc_date:%B} {utc_date.day}, {utc_date.year}"


def _write_volume_html(
    destination: Path,
    *,
    label: str,
    conversations: Sequence[Conversation],
    options: CompileOptions,
    redactors: Sequence[tuple[re.Pattern[str], str]],
) -> int:
    """Stream one self-contained, print-ready HTML volume and return rendered message count."""

    entries = [
        (index, conversation, _visible_messages(conversation, options))
        for index, conversation in enumerate(conversations, start=1)
    ]
    rendered_message_count = sum(len(messages) for _, _, messages in entries)
    escaped_title = html.escape(_redact(options.title, redactors))
    volume_title = (
        escaped_title if label == "complete" else f"{escaped_title} — {html.escape(label)}"
    )

    with _atomic_text_writer(destination) as handle:
        handle.write('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n')
        handle.write(
            f"<title>{volume_title}</title>\n<style>{_DOCUMENT_CSS}</style></head><body>\n"
        )
        handle.write('<section class="cover">')
        handle.write(f"<h1>{escaped_title}</h1>")
        if options.subtitle:
            handle.write(
                f'<p class="subtitle">{html.escape(_redact(options.subtitle, redactors))}</p>'
            )
        if label != "complete":
            handle.write(f'<p class="volume-label">Volume {html.escape(label)}</p>')
        if options.author:
            handle.write(f'<p class="author">{html.escape(_redact(options.author, redactors))}</p>')
        handle.write("</section>\n")
        handle.write('<section class="front-matter"><h1>About this volume</h1>')
        handle.write(
            f"<p>{len(entries):,} conversations and {rendered_message_count:,} visible-path "
            "messages. Alternate branches and non-user-facing records remain preserved in the "
            "Archive IR but are not merged into these transcripts.</p>"
        )
        if not options.include_reasoning_summaries:
            handle.write("<p>Thinking traces and reasoning summaries are omitted.</p>")
        handle.write('</section>\n<section class="toc"><h1>Contents</h1><ol>')
        for index, conversation, _ in entries:
            title = html.escape(_redact(conversation.title or "Untitled conversation", redactors))
            handle.write(
                f'<li><a href="#conversation-{index:06d}">{title}</a>'
                f"<span>{html.escape(_display_date(conversation))}</span></li>"
            )
        handle.write("</ol></section>\n<main>\n")
        for index, conversation, messages in entries:
            title = html.escape(_redact(conversation.title or "Untitled conversation", redactors))
            handle.write(
                f'<article class="conversation" id="conversation-{index:06d}">'
                f"<header><p>{html.escape(_display_date(conversation))}</p><h1>{title}</h1></header>"
            )
            if not messages:
                handle.write('<p class="empty">No user-facing current-path messages.</p>')
            for message in messages:
                role = html.escape(message.role.value.title())
                handle.write(f'<section class="message {message.role.value}"><h2>{role}</h2>')
                if message.created_at is not None:
                    timestamp = message.created_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
                    handle.write(f'<p class="timestamp">{timestamp}</p>')
                for block in message.content:
                    handle.write(_render_block(block, options=options, redactors=redactors))
                handle.write("</section>")
            handle.write("</article>\n")
        handle.write("</main></body></html>\n")
    return rendered_message_count


def _write_index_html(
    destination: Path,
    *,
    options: CompileOptions,
    analysis: CorpusAnalysis,
    volumes: Sequence[CompiledVolume],
    redactors: Sequence[tuple[re.Pattern[str], str]],
) -> Path:
    """Write a compact local landing page linking every compiled volume."""

    with _atomic_text_writer(destination) as handle:
        title = html.escape(_redact(options.title, redactors))
        handle.write(
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>{title}</title><style>{_INDEX_CSS}</style></head><body><main>"
            f"<h1>{title}</h1><p>{analysis.conversation_count:,} source conversations; "
            f"{analysis.current_path_message_count:,} current-path messages.</p><ul>"
        )
        for volume in volumes:
            html_name = html.escape(volume.html_path.name)
            volume_label = html.escape(volume.label)
            handle.write(
                f'<li><a href="{html_name}">{volume_label}</a> '
                f"— {volume.conversation_count:,} conversations"
            )
            if volume.pdf_path is not None:
                handle.write(f' · <a href="{html.escape(volume.pdf_path.name)}">PDF</a>')
            handle.write("</li>")
        handle.write("</ul></main></body></html>\n")
    return destination


def _offline_url_fetcher(url: str, *args: object, **kwargs: object) -> object:
    """Reject every subsidiary URL load during chronological PDF generation."""

    del args, kwargs
    scheme = urlsplit(url).scheme.casefold() or "local"
    raise ArchiveCompilationError(f"Chronological PDF blocked a {scheme!r} asset request.")


def _render_pdf(html_path: Path, destination: Path) -> Path:
    """Render local HTML to PDF with every subsidiary-resource request disabled."""

    try:
        weasyprint = importlib.import_module("weasyprint")
    except ImportError as exc:
        raise ArchiveCompilationError(
            "PDF rendering requires the optional 'pdf' dependency set."
        ) from exc

    temporary_path = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        html_factory = cast(_HTMLFactory, weasyprint.HTML)
        html_factory(
            filename=str(html_path),
            base_url=str(html_path.parent),
            url_fetcher=_offline_url_fetcher,
        ).write_pdf(str(temporary_path))
        os.replace(temporary_path, destination)
    except Exception as exc:
        cause_name = type(exc).__name__
        raise ArchiveCompilationError(
            f"PDF rendering failed safely ({cause_name}): {destination}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def compile_archive(
    archive: Archive,
    output_directory: str | Path,
    *,
    options: CompileOptions | None = None,
) -> CompilationResult:
    """Analyze and compile one Archive IR into self-contained HTML and optional PDF volumes.

    Only declared current paths are rendered. Compilation is local and never embeds or fetches
    remote assets. Private artifacts are written beneath ``output_directory``; callers should keep
    that directory out of Git history.
    """

    active_options = options or CompileOptions()
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    analysis = analyze_archive(archive)
    analysis_path = _write_json(analysis, destination / "analysis.json")
    redactors = _compile_redactors(active_options)

    selected = sorted(archive.conversations, key=_conversation_sort_key)
    if active_options.max_conversations is not None:
        selected = selected[: active_options.max_conversations]
    grouped = _group_conversations(selected, active_options)
    volumes: list[CompiledVolume] = []
    for label, conversations in grouped:
        stem = "archive" if label == "complete" else f"archive-{label}"
        html_path = destination / f"{stem}.html"
        message_count = _write_volume_html(
            html_path,
            label=label,
            conversations=conversations,
            options=active_options,
            redactors=redactors,
        )
        pdf_path = destination / f"{stem}.pdf" if active_options.render_pdf else None
        if pdf_path is not None:
            _render_pdf(html_path, pdf_path)
        volumes.append(
            CompiledVolume(
                label=label,
                conversation_count=len(conversations),
                message_count=message_count,
                html_path=html_path,
                html_sha256=_sha256(html_path),
                pdf_path=pdf_path,
                pdf_sha256=_sha256(pdf_path) if pdf_path is not None else None,
            )
        )

    index_path = _write_index_html(
        destination / "index.html",
        options=active_options,
        analysis=analysis,
        volumes=volumes,
        redactors=redactors,
    )
    manifest_path = destination / "compilation_manifest.json"
    result = CompilationResult(
        output_directory=destination,
        analysis_path=analysis_path,
        index_path=index_path,
        manifest_path=manifest_path,
        selected_conversation_count=len(selected),
        volumes=tuple(volumes),
    )
    _write_json(
        {
            "compiler_version": "1.1",
            "package_version": __version__,
            "archive_version": archive.archive_version,
            "source_archive_sha256": archive.source_manifest.archive_sha256,
            "selected_conversation_count": len(selected),
            "volume_mode": active_options.volume_mode.value,
            "reasoning_summaries_included": active_options.include_reasoning_summaries,
            "system_messages_included": active_options.include_system_messages,
            "tool_messages_included": active_options.include_tool_messages,
            "redaction_rule_count": len(active_options.redaction_rules),
            "analysis": analysis_path.name,
            "index": index_path.name,
            "volumes": [
                {
                    "label": volume.label,
                    "conversation_count": volume.conversation_count,
                    "message_count": volume.message_count,
                    "html": volume.html_path.name,
                    "html_sha256": volume.html_sha256,
                    "pdf": volume.pdf_path.name if volume.pdf_path is not None else None,
                    "pdf_sha256": volume.pdf_sha256,
                }
                for volume in volumes
            ],
        },
        manifest_path,
    )
    return result


_DOCUMENT_CSS = r"""
@page {
  size: A4;
  margin: 22mm 19mm 24mm;
  @bottom-center { content: counter(page); color: #667085; font-size: 9pt; }
}
@page :first { @bottom-center { content: none; } }
:root {
  color: #17202a;
  font-family: Inter, "Source Sans 3", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.55;
}
* { box-sizing: border-box; }
body { margin: 0; background: #fff; }
a { color: #175cd3; text-decoration: none; }
.cover {
  break-after: page;
  min-height: 245mm;
  display: flex;
  flex-direction: column;
  justify-content: center;
  border-top: 8px solid #175cd3;
}
.cover h1 {
  max-width: 150mm;
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 34pt;
  line-height: 1.08;
  color: #101828;
}
.subtitle { max-width: 140mm; margin-top: 12mm; color: #475467; font-size: 16pt; }
.volume-label, .author {
  margin-top: 18mm;
  color: #175cd3;
  font-size: 12pt;
  font-weight: 600;
  letter-spacing: .08em;
  text-transform: uppercase;
}
.front-matter, .toc { break-after: page; }
.front-matter h1, .toc h1 {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 25pt;
  color: #101828;
}
.toc ol { padding: 0; list-style: none; }
.toc li {
  display: flex;
  gap: 8mm;
  justify-content: space-between;
  padding: 2.2mm 0;
  border-bottom: .2mm solid #e4e7ec;
}
.toc li span { white-space: nowrap; color: #667085; }
.conversation { break-before: page; }
.conversation > header { padding-bottom: 8mm; border-bottom: 1mm solid #175cd3; }
.conversation > header p {
  margin: 0 0 2mm;
  color: #667085;
  font-size: 9pt;
  text-transform: uppercase;
  letter-spacing: .06em;
}
.conversation > header h1 {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 23pt;
  line-height: 1.15;
  color: #101828;
}
.message { margin: 8mm 0; padding: 5mm 6mm; border-radius: 3mm; break-inside: auto; }
.message.user { background: #eff8ff; border-left: 1.2mm solid #2e90fa; }
.message.assistant { background: #f9fafb; border-left: 1.2mm solid #98a2b3; }
.message.system, .message.developer, .message.tool {
  background: #fff7ed;
  border-left: 1.2mm solid #f79009;
}
.message h2 {
  margin: 0 0 3mm;
  color: #344054;
  font-size: 9pt;
  letter-spacing: .08em;
  text-transform: uppercase;
}
.message p { margin: 0 0 3.5mm; }
.timestamp { float: right; margin-top: -8mm !important; color: #98a2b3; font-size: 8pt; }
pre {
  overflow-wrap: anywhere;
  white-space: pre-wrap;
  padding: 4mm;
  border-radius: 2mm;
  background: #101828;
  color: #f2f4f7;
  font: 8.5pt/1.45 "Source Code Pro", Consolas, monospace;
}
code { overflow-wrap: anywhere; font-family: "Source Code Pro", Consolas, monospace; }
.code-label { margin: 3mm 0 -2mm; color: #667085; font-size: 8pt; text-transform: uppercase; }
.reasoning {
  margin: 4mm 0;
  padding: 4mm;
  border: .3mm dashed #98a2b3;
  color: #475467;
  background: #fcfcfd;
}
.reasoning strong {
  display: block;
  margin-bottom: 2mm;
  font-size: 8.5pt;
  text-transform: uppercase;
}
.reference, .unsupported, .empty { color: #667085; font-style: italic; }
blockquote { margin: 4mm 0; padding-left: 5mm; border-left: 1mm solid #d0d5dd; color: #475467; }
table { width: 100%; border-collapse: collapse; margin: 4mm 0; font-size: 9pt; }
th, td { padding: 2mm; border: .2mm solid #d0d5dd; vertical-align: top; }
"""

_INDEX_CSS = r"""
:root { color: #17202a; font-family: Inter, "Helvetica Neue", Arial, sans-serif; line-height: 1.5; }
body { margin: 0; background: #f2f4f7; }
main {
  max-width: 760px;
  margin: 48px auto;
  padding: 40px;
  border-radius: 18px;
  background: #fff;
  box-shadow: 0 12px 40px rgba(16,24,40,.08);
}
h1 { margin-top: 0; font: 700 2.2rem/1.15 Georgia, serif; }
ul { padding: 0; list-style: none; }
li { padding: 12px 0; border-bottom: 1px solid #e4e7ec; }
a { color: #175cd3; font-weight: 600; text-decoration: none; }
"""
