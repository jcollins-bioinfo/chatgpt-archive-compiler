"""Private Parquet, JSON, Markdown, and navigable HTML semantic-atlas exports."""

# ruff: noqa: E501

from __future__ import annotations

import html
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import IO

from pydantic import BaseModel

from chatgpt_archive_compiler.semantic.errors import SemanticAtlasError
from chatgpt_archive_compiler.semantic.models import (
    ArchiveSynthesis,
    Category,
    CategoryProfile,
    ProjectTimeline,
    SemanticAtlas,
)


@contextmanager
def _atomic_text_writer(path: Path) -> Iterator[IO[str]]:
    """Yield a UTF-8 temporary writer and atomically replace its destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise SemanticAtlasError("Could not write a semantic-atlas artifact.") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: BaseModel | Mapping[str, object] | Sequence[object]) -> Path:
    """Write one canonical JSON semantic artifact."""

    data: object = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    with _atomic_text_writer(path) as handle:
        json.dump(data, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write stable Zstandard-compressed Parquet through the required Polars dependency."""

    try:
        polars = import_module("polars")
    except ImportError as exc:
        raise SemanticAtlasError(
            "Parquet export requires the project's Polars dependency."
        ) from exc
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame = polars.DataFrame(rows)
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        os.replace(temporary, path)
    except Exception as exc:
        raise SemanticAtlasError(f"Parquet export failed safely ({type(exc).__name__}).") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _catalog_rows(atlas: SemanticAtlas) -> list[dict[str, object]]:
    """Join representations, profiles, and taxonomy assignments into catalog rows."""

    analyses = {item.conversation_key: item for item in atlas.analyses}
    assignments = {item.conversation_key: item for item in atlas.taxonomy.assignments}
    rows: list[dict[str, object]] = []
    for representation in atlas.representations:
        analysis = analyses[representation.conversation_key]
        assignment = assignments[representation.conversation_key]
        rows.append(
            {
                "conversation_key": representation.conversation_key,
                "source_index": representation.source_index,
                "source_conversation_id": representation.source_conversation_id,
                "title": representation.title,
                "created_at": representation.created_at,
                "updated_at": representation.updated_at,
                "message_count": representation.message_count,
                "character_count": representation.character_count,
                "original_character_count": representation.original_character_count,
                "truncated": representation.truncated,
                "text_sha256": representation.text_sha256,
                "synopsis": analysis.synopsis,
                "primary_subject": analysis.primary_subject,
                "secondary_subjects": list(analysis.secondary_subjects),
                "projects": list(analysis.projects),
                "conversation_type": analysis.conversation_type,
                "entities": list(analysis.entities),
                "goals": list(analysis.goals),
                "decisions": list(analysis.decisions),
                "unresolved_questions": list(analysis.unresolved_questions),
                "recurring_themes": list(analysis.recurring_themes),
                "temporal_role": analysis.temporal_role,
                "related_conversation_keys": list(analysis.related_conversation_keys),
                "analysis_confidence": analysis.confidence,
                "primary_category_id": assignment.primary_category_id,
                "secondary_category_ids": list(assignment.secondary_category_ids),
                "category_confidence": assignment.confidence,
            }
        )
    return rows


def _markdown_list(values: Sequence[str]) -> str:
    """Render a Markdown list or an explicit no-findings sentence."""

    if not values:
        return "No strong archive-level finding was recorded.\n"
    return "\n".join(f"- {_escape_markdown(value)}" for value in values) + "\n"


def _escape_markdown(value: str) -> str:
    """Escape model-derived Markdown/HTML control characters while preserving readable text."""

    escaped = value.replace("\\", "\\\\")
    for character in "`*_{}[]<>()#+-.!|":
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


def _write_synthesis_markdown(path: Path, synthesis: ArchiveSynthesis) -> Path:
    """Write a readable, model-independent global synthesis document."""

    sections = (
        ("Dominant domains", synthesis.dominant_domains),
        ("Cross-domain connections", synthesis.cross_domain_connections),
        ("Recurring patterns", synthesis.recurring_patterns),
        ("Evolution over time", synthesis.temporal_evolution),
        ("Tensions and reversals", synthesis.tensions_and_reversals),
        ("Dormant threads", synthesis.dormant_threads),
        ("Opportunities", synthesis.opportunities),
    )
    with _atomic_text_writer(path) as handle:
        handle.write("# Cross-archive synthesis\n\n")
        handle.write(_escape_markdown(synthesis.executive_summary.strip()))
        handle.write("\n\n")
        for heading, values in sections:
            handle.write(f"## {heading}\n\n")
            handle.write(_markdown_list(values))
            handle.write("\n")
    return path


def _date_label(value: datetime | None) -> str:
    """Format an optional timestamp as a compact UTC calendar label."""

    return value.astimezone(UTC).strftime("%b %Y") if value is not None else "Undated"


def _html_list(values: Sequence[str], *, empty: str = "No strong finding recorded.") -> str:
    """Render source-derived strings as safely escaped semantic HTML."""

    if not values:
        return f'<p class="muted">{html.escape(empty)}</p>'
    return "<ul>" + "".join(f"<li>{html.escape(value)}</li>" for value in values) + "</ul>"


def _category_section(
    category: Category,
    *,
    profile: CategoryProfile | None,
    title_by_key: Mapping[str, str],
) -> str:
    """Render one leaf category and representative conversations."""

    body = [
        f'<article class="category" id="{html.escape(category.category_id)}">',
        '<div class="eyebrow">Discovered category</div>',
        f"<h3>{html.escape(category.name)}</h3>",
        f"<p>{html.escape(category.description)}</p>",
        '<div class="chips">',
        *(f"<span>{html.escape(concept)}</span>" for concept in category.defining_concepts[:8]),
        "</div>",
    ]
    if profile is not None:
        body.extend(
            (
                f'<p class="profile">{html.escape(profile.overview)}</p>',
                "<h4>Characteristic questions</h4>",
                _html_list(profile.characteristic_questions),
                "<h4>Major conclusions</h4>",
                _html_list(profile.major_conclusions),
                "<h4>Open threads</h4>",
                _html_list(profile.unresolved_threads),
            )
        )
        representatives = profile.representative_conversation_keys
    else:
        representatives = category.conversation_keys[:5]
    if representatives:
        body.append('<h4>Representative conversations</h4><ol class="conversations">')
        body.extend(
            f"<li>{html.escape(title_by_key.get(key, key))}</li>" for key in representatives
        )
        body.append("</ol>")
    body.append("</article>")
    return "".join(body)


def _timeline_section(timeline: ProjectTimeline) -> str:
    """Render one recurring project timeline with safe local anchors."""

    events = "".join(
        "<li>"
        f"<time>{html.escape(_date_label(event.occurred_at))}</time>"
        f"<div><strong>{html.escape(event.label)}</strong><p>{html.escape(event.summary)}</p></div>"
        "</li>"
        for event in timeline.events
    )
    return (
        '<article class="timeline">'
        f"<h3>{html.escape(timeline.project_name)}</h3>"
        f"<p>{html.escape(timeline.overview)}</p>"
        f"<ol>{events}</ol>"
        + (
            f'<p class="current"><strong>Current state.</strong> '
            f"{html.escape(timeline.current_state)}</p>"
            if timeline.current_state
            else ""
        )
        + "</article>"
    )


def _write_atlas_html(path: Path, atlas: SemanticAtlas) -> Path:
    """Write a polished, print-conscious, navigable semantic atlas."""

    title_by_key = {item.conversation_key: item.title for item in atlas.representations}
    profiles = {profile.category_id: profile for profile in atlas.category_profiles}
    categories = {category.category_id: category for category in atlas.taxonomy.categories}
    parents = sorted(
        (category for category in atlas.taxonomy.categories if category.parent_id is None),
        key=lambda category: category.name.casefold(),
    )
    navigation = "".join(
        f'<a href="#{html.escape(category.category_id)}">{html.escape(category.name)}</a>'
        for category in parents
    )
    domain_sections: list[str] = []
    for parent in parents:
        leaves = [categories[child_id] for child_id in parent.child_ids if child_id in categories]
        domain_sections.append(
            f'<section class="domain" id="{html.escape(parent.category_id)}">'
            '<header class="domain-title"><div class="eyebrow">Domain</div>'
            f"<h2>{html.escape(parent.name)}</h2><p>{html.escape(parent.description)}</p></header>"
            + "".join(
                _category_section(
                    leaf,
                    profile=profiles.get(leaf.category_id),
                    title_by_key=title_by_key,
                )
                for leaf in leaves
            )
            + "</section>"
        )
    synthesis = atlas.synthesis
    synthesis_cards = "".join(
        f"<article><h3>{html.escape(heading)}</h3>{_html_list(values)}</article>"
        for heading, values in (
            ("Cross-domain connections", synthesis.cross_domain_connections),
            ("Recurring patterns", synthesis.recurring_patterns),
            ("Evolution over time", synthesis.temporal_evolution),
            ("Tensions and reversals", synthesis.tensions_and_reversals),
            ("Dormant threads", synthesis.dormant_threads),
            ("Opportunities", synthesis.opportunities),
        )
    )
    timelines = "".join(_timeline_section(item) for item in atlas.project_timelines)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic Atlas of Conversations</title><style>{_ATLAS_CSS}</style></head>
<body><header class="masthead"><div class="edition">PRIVATE ARCHIVE · SEMANTIC EDITION</div>
<h1>Semantic Atlas<br><em>of Conversations</em></h1>
<p class="dek">A thematic, relational, and chronological reading of
{len(atlas.representations):,} conversations—organized by meaning rather than storage order.</p>
<div class="rule"></div><p class="summary">{html.escape(synthesis.executive_summary)}</p></header>
<nav>{navigation}</nav><main>
<section class="synthesis"><header><div class="eyebrow">Overarching analysis</div>
<h2>Patterns across the archive</h2></header><div class="synthesis-grid">{synthesis_cards}</div></section>
<section class="projects"><header><div class="eyebrow">Longitudinal view</div>
<h2>Projects through time</h2></header>{timelines or '<p class="muted">No recurring project timeline was produced.</p>'}</section>
{''.join(domain_sections)}
</main><footer>Generated locally from normalized Archive IR. Source transcripts remain private.</footer>
</body></html>
"""
    with _atomic_text_writer(path) as handle:
        handle.write(document)
    return path


def write_semantic_artifacts(atlas: SemanticAtlas, output_directory: Path) -> dict[str, Path]:
    """Export the complete atlas into interoperable analytical and reading formats."""

    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "catalog": _write_parquet(
            output_directory / "conversation_catalog.parquet", _catalog_rows(atlas)
        ),
        "embeddings": _write_parquet(
            output_directory / "embeddings.parquet",
            [
                {
                    "conversation_key": record.conversation_key,
                    "vector": list(record.vector),
                }
                for record in atlas.embeddings
            ],
        ),
        "graph": _write_parquet(
            output_directory / "semantic_graph.parquet",
            [
                {
                    "source_key": edge.source_key,
                    "target_key": edge.target_key,
                    "similarity": edge.similarity,
                    "relation_type": edge.relation_type,
                    "shared_signals": list(edge.shared_signals),
                }
                for edge in atlas.graph_edges
            ],
        ),
        "taxonomy": _write_json(output_directory / "taxonomy.json", atlas.taxonomy),
        "category_profiles": _write_json(
            output_directory / "category_profiles.json",
            [profile.model_dump(mode="json") for profile in atlas.category_profiles],
        ),
        "project_timelines": _write_json(
            output_directory / "project_timelines.json",
            [timeline.model_dump(mode="json") for timeline in atlas.project_timelines],
        ),
        "synthesis": _write_synthesis_markdown(
            output_directory / "cross_archive_synthesis.md", atlas.synthesis
        ),
        "review_queue": _write_json(output_directory / "review_queue.json", atlas.review_queue),
        "atlas_html": _write_atlas_html(output_directory / "semantic_atlas.html", atlas),
    }
    return paths


_ATLAS_CSS = r"""
@page { size: A4; margin: 22mm 19mm 24mm; @bottom-center { content: counter(page); color: #718078; font: 9pt sans-serif; } }
@page :first { @bottom-center { content: none; } }
:root { --ink:#17231f; --muted:#617169; --paper:#f7f5ee; --card:#fffefa; --line:#cdd6cf; --moss:#315e4b; --gold:#a9782b; color:var(--ink); background:var(--paper); font:17px/1.65 Inter,"Source Sans 3","Helvetica Neue",sans-serif; }
* { box-sizing:border-box; }
body { margin:0; }
.masthead { min-height:92vh; padding:10vh max(7vw,28px); display:flex; flex-direction:column; justify-content:center; background:linear-gradient(135deg,#f7f5ee 0%,#edf2eb 100%); }
.edition,.eyebrow { color:var(--gold); font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
h1,h2,h3 { font-family:"Iowan Old Style",Charter,"Source Serif 4",Georgia,serif; font-weight:500; }
h1 { max-width:1000px; margin:.7rem 0 1.4rem; font-size:clamp(3.6rem,9vw,8rem); line-height:.85; letter-spacing:-.055em; }
h1 em { color:var(--moss); font-weight:400; }
.dek { max-width:780px; color:var(--muted); font:clamp(1.15rem,2.2vw,1.55rem)/1.5 "Iowan Old Style",Charter,Georgia,serif; }
.rule { width:110px; height:4px; margin:2.5rem 0; background:var(--gold); }
.summary { max-width:880px; font-size:1.05rem; }
nav { position:sticky; top:0; z-index:3; display:flex; gap:1.2rem; overflow:auto; padding:1rem max(4vw,20px); border-block:1px solid var(--line); background:rgba(247,245,238,.94); backdrop-filter:blur(14px); }
nav a { flex:none; color:var(--moss); font-size:.82rem; font-weight:600; text-decoration:none; }
main { max-width:1180px; margin:auto; padding:5rem max(4vw,22px); }
section>header>h2,.domain-title h2 { max-width:850px; margin:.4rem 0 1rem; font-size:clamp(2.6rem,5vw,4.8rem); line-height:1; letter-spacing:-.035em; }
.synthesis,.projects,.domain { margin-bottom:8rem; scroll-margin-top:5rem; }
.synthesis-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:1rem; margin-top:2.5rem; }
.synthesis-grid article,.category,.timeline { padding:clamp(1.4rem,3vw,2.4rem); border:1px solid var(--line); border-radius:6px; background:var(--card); box-shadow:0 14px 36px rgba(23,35,31,.05); }
.synthesis-grid h3,.category h3,.timeline h3 { margin:.2rem 0 .7rem; color:var(--moss); font-size:1.55rem; line-height:1.15; }
.domain-title { margin-bottom:2rem; padding-bottom:2rem; border-bottom:3px double var(--line); }
.category { margin:1.2rem 0; break-inside:avoid; }
.category h4 { margin:1.5rem 0 .25rem; color:var(--gold); font-size:.75rem; letter-spacing:.1em; text-transform:uppercase; }
.chips { display:flex; flex-wrap:wrap; gap:.45rem; margin:1rem 0; }
.chips span { padding:.2rem .65rem; border-radius:99px; background:#e8eee9; color:var(--moss); font-size:.76rem; }
.profile { padding-left:1rem; border-left:3px solid var(--gold); }
.muted { color:var(--muted); font-style:italic; }
.timeline { margin:1rem 0; }
.timeline ol { padding:0; list-style:none; }
.timeline li { display:grid; grid-template-columns:110px 1fr; gap:1.2rem; padding:1rem 0; border-top:1px solid var(--line); }
.timeline time { color:var(--gold); font-size:.8rem; font-weight:700; text-transform:uppercase; }
.timeline p { margin:.25rem 0; }
footer { padding:3rem; border-top:1px solid var(--line); color:var(--muted); text-align:center; font-size:.8rem; }
@media(max-width:620px){ .timeline li{grid-template-columns:1fr;gap:.2rem} main{padding-top:3rem} }
@media print { :root{font-size:10pt;background:#fff} nav{display:none}.masthead{min-height:250mm;break-after:page;padding:20mm 0}.synthesis,.projects,.domain{break-before:page;margin:0}.category,.timeline,.synthesis-grid article{box-shadow:none}.category{break-inside:auto} a{color:inherit} }
"""
