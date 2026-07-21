"""Dash application factory with isolated local state and thin callbacks."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from dash import Dash, Input, Output, State, dcc, html, no_update

from chatgpt_archive_compiler.app.downloads import make_download_payload
from chatgpt_archive_compiler.app.jobs import JobRegistry, JobStatus
from chatgpt_archive_compiler.app.service import (
    STAGES,
    compile_for_app,
    preflight_archive,
)
from chatgpt_archive_compiler.app.state import SessionStore
from chatgpt_archive_compiler.app.visualizations import (
    build_figures,
    load_dashboard_data,
)
from chatgpt_archive_compiler.version import __version__

MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def _aria_props(**attributes: str) -> dict[str, Any]:
    """Convert Python-safe ARIA names into Dash HTML attribute names."""

    return {name.replace("_", "-"): value for name, value in attributes.items()}


def _layout() -> html.Main:
    return html.Main(
        [
            dcc.Store(id="session-id"),
            dcc.Store(id="job-id"),
            dcc.Interval(id="job-poll", interval=750, disabled=True),
            html.Header(
                [
                    html.P(
                        "LOCAL-FIRST · EVIDENCE-LINKED · VERIFIABLE",
                        className="eyebrow",
                    ),
                    html.H1("Turn conversation history into a knowledge atlas."),
                    html.P(
                        "Upload your ChatGPT export and compile years of conversations into a "
                        "private, evidence-linked map of projects, ideas, decisions, recurring "
                        "themes, and intellectual development.",
                        className="lede",
                    ),
                ],
                className="hero",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.H2("1 · Choose your archive"),
                            dcc.Upload(
                                id="upload-export-zip",
                                children=html.Div(
                                    [
                                        html.Strong("Drop your export ZIP here"),
                                        html.Span(" or choose a file"),
                                    ]
                                ),
                                accept=".zip,application/zip",
                                multiple=False,
                                className="upload",
                            ),
                            html.P(
                                "Maximum 512 MiB. The archive remains on this computer.",
                                className="hint",
                            ),
                            html.Div(id="preflight", className="panel muted"),
                        ],
                        className="card",
                    ),
                    html.Div(
                        [
                            html.H2("2 · Privacy boundary"),
                            dcc.RadioItems(
                                id="processing-mode",
                                options=[
                                    {
                                        "label": "Local deterministic baseline (recommended)",
                                        "value": "local",
                                    },
                                    {
                                        "label": "OpenAI-enhanced (requires explicit consent)",
                                        "value": "openai",
                                        "disabled": True,
                                    },
                                ],
                                value="local",
                            ),
                            dcc.Checklist(
                                id="professional-safe",
                                options=[
                                    {
                                        "label": " Professional-safe mode",
                                        "value": "enabled",
                                    }
                                ],
                                value=["enabled"],
                            ),
                            html.P(
                                "Exclude conversations that may be too private, intimate, "
                                "sensitive, "
                                "or professionally inappropriate for a public-facing archive."
                            ),
                            html.P(
                                "Review items are excluded until approved. Automated filtering can "
                                "produce false positives and false negatives. Review the "
                                "Professional-safe output before publishing or sharing it.",
                                className="warning",
                            ),
                            html.Button(
                                "Compile Archive",
                                id="compile",
                                disabled=True,
                                className="primary",
                            ),
                        ],
                        className="card",
                    ),
                ],
                className="grid",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.H2("Interactive knowledge atlas"),
                            dcc.Checklist(
                                id="anonymous-mode",
                                options=[
                                    {
                                        "label": " Anonymous visualization mode",
                                        "value": "anonymous",
                                    }
                                ],
                                value=[],
                                className="privacy-toggle",
                            ),
                            html.P(
                                "Presentation-oriented suppression is not guaranteed "
                                "de-identification.",
                                className="hint",
                            ),
                        ],
                        className="results-heading",
                    ),
                    html.Div(
                        id="atlas-results",
                        children="Compile an archive to reveal its structure.",
                    ),
                ],
                className="card results-card",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.H2("Compilation progress"),
                            html.Div(
                                id="progress",
                                children="Waiting for a safe archive preflight.",
                            ),
                        ],
                        className="card",
                    ),
                    html.Div(
                        [
                            html.H2("Semantic overview"),
                            html.Div(
                                id="overview",
                                children=(
                                    "Projects, themes, open loops, evidence, and provenance will "
                                    "appear here."
                                ),
                            ),
                        ],
                        className="card",
                    ),
                ],
                className="grid",
            ),
            html.Section(
                [
                    html.H2("Evidence & export center"),
                    html.Div(
                        id="exports",
                        children="Verified artifacts are available after compilation.",
                    ),
                    dcc.Download(id="download-bundle"),
                ],
                className="card",
            ),
            html.Footer(
                f"ChatGPT Archive Compiler {__version__} · No telemetry · No network in local mode"
            ),
        ]
    )


def create_app(*, data_root: Path | None = None) -> Dash:
    """Create an isolated Dash app; no mutable module-global user state is used."""

    root = data_root or Path(os.environ.get("CHATGPT_ARCHIVE_OUTPUT", "./archive-output"))
    sessions = SessionStore(root / "sessions")
    jobs = JobRegistry()
    app = Dash(__name__, title="Private Semantic Atlas", suppress_callback_exceptions=True)
    app.server.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.layout = _layout

    @app.callback(
        Output("session-id", "data"),
        Output("preflight", "children"),
        Output("compile", "disabled"),
        Input("upload-export-zip", "contents"),
        State("upload-export-zip", "filename"),
        prevent_initial_call=True,
    )
    def upload(contents: str | None, filename: str | None):  # type: ignore[no-untyped-def]
        if not contents or not filename:
            return no_update, "No archive received.", True
        try:
            encoded = contents.split(",", 1)[1]
            payload = base64.b64decode(encoded, validate=True)
            if len(payload) > MAX_UPLOAD_BYTES:
                raise ValueError("upload limit")
            session_id = sessions.create()
            path = sessions.path(session_id) / "source.zip"
            path.write_bytes(payload)
            summary = preflight_archive(path)
        except Exception as exception:
            return (
                no_update,
                f"Preflight failed safely ({type(exception).__name__}).",
                True,
            )
        return (
            session_id,
            html.Ul(
                [
                    html.Li(f"File: {Path(filename).name}"),
                    html.Li(f"Size: {summary.size_bytes:,} bytes"),
                    html.Li(f"ZIP safety: passed · {summary.member_count} members"),
                    html.Li(
                        "Expected conversation member: "
                        f"{'found' if summary.conversation_member_found else 'not found'}"
                    ),
                    html.Li(f"Session: {session_id[:8]}…"),
                ]
            ),
            not summary.conversation_member_found,
        )

    @app.callback(
        Output("job-id", "data"),
        Output("job-poll", "disabled"),
        Output("progress", "children", allow_duplicate=True),
        Input("compile", "n_clicks"),
        State("session-id", "data"),
        State("professional-safe", "value"),
        prevent_initial_call=True,
    )
    def start(_clicks: int, session_id: str | None, safety: list[str]):  # type: ignore[no-untyped-def]
        if not session_id:
            return no_update, True, "Upload and preflight an archive first."
        session_path = sessions.path(session_id)
        job_id = jobs.submit(
            lambda stage: compile_for_app(
                session_path / "source.zip",
                session_path / "output",
                professional_safe="enabled" in safety,
                stage_callback=stage,
            )
        )
        return job_id, False, "Queued on the local worker…"

    @app.callback(
        Output("progress", "children"),
        Output("overview", "children"),
        Output("exports", "children"),
        Output("atlas-results", "children"),
        Output("job-poll", "disabled", allow_duplicate=True),
        Input("job-poll", "n_intervals"),
        Input("anonymous-mode", "value"),
        State("job-id", "data"),
        prevent_initial_call=True,
    )
    def poll(  # type: ignore[no-untyped-def]
        _ticks: int, anonymous_values: list[str], job_id: str | None
    ):
        job = jobs.get(job_id or "")
        if job is None:
            return "Unknown local job.", no_update, no_update, no_update, True
        elapsed = (
            0
            if job.started_at is None
            else (job.finished_at or __import__("time").monotonic()) - job.started_at
        )
        completed = set(job.completed_stages)
        stages = list(STAGES)
        orbit = html.Ol(
            [
                html.Li(
                    html.Span(stage),
                    className=(
                        "orbit-node complete"
                        if stage in completed
                        else ("orbit-node active" if stage == job.stage else "orbit-node future")
                    ),
                    **_aria_props(aria_current="step" if stage == job.stage else "false"),
                )
                for stage in dict.fromkeys(stages)
            ],
            className="pipeline-orbit",
            **_aria_props(aria_label="Semantic compilation stages"),
        )
        log = html.Details(
            [
                html.Summary(f"Execution log · {len(job.log)} entries"),
                html.Ol(
                    [
                        html.Li(
                            [
                                html.Time(
                                    f"{int(entry.elapsed_seconds) // 60:02d}:"
                                    f"{int(entry.elapsed_seconds) % 60:02d}"
                                ),
                                html.Strong(entry.status.value.upper()),
                                html.Span(entry.stage),
                                html.Small(entry.description),
                            ],
                            className=f"log-entry {entry.status.value}",
                        )
                        for entry in job.log
                    ],
                    className="execution-log",
                ),
            ],
            open=job.status is not JobStatus.SUCCEEDED,
        )
        progress = html.Div(
            [
                html.P(
                    f"{job.stage} · {len(completed)} stages completed · {elapsed:.1f}s elapsed",
                    role="status",
                    **_aria_props(aria_live="polite"),
                ),
                orbit,
                log,
            ]
        )
        if job.status is JobStatus.FAILED:
            return job.error, no_update, no_update, no_update, True
        if job.status is not JobStatus.SUCCEEDED:
            return progress, no_update, no_update, no_update, False
        result = job.result
        overview = html.Div(
            [
                html.Div(
                    [
                        html.Strong(str(result.project_count)),
                        html.Span(" project threads"),
                    ],
                    className="metric",
                ),
                html.Div(
                    [
                        html.Strong(str(result.category_count)),
                        html.Span(" hierarchical categories"),
                    ],
                    className="metric",
                ),
                html.Div(
                    [
                        html.Strong(str(result.included_count)),
                        html.Span(" approved conversations"),
                    ],
                    className="metric",
                ),
                html.Div(
                    [
                        html.Strong(str(result.excluded_count + result.review_count)),
                        html.Span(" excluded or awaiting review"),
                    ],
                    className="metric",
                ),
                html.P(
                    "Select the semantic atlas in the verified bundle to inspect categories, "
                    "timelines, review signals, and source-conversation evidence."
                ),
            ],
            className="metrics",
        )
        exports = html.Div(
            [
                html.P("Verification manifest and SHA-256 inventory complete."),
                html.Button(
                    "Download professional-safe verified bundle",
                    id="download-button",
                    className="primary",
                ),
            ]
        )
        data = load_dashboard_data(result.dashboard_data_path)
        anonymous = "anonymous" in anonymous_values
        figures = build_figures(data, anonymous=anonymous)
        synthesis = data["synthesis"]
        summary_items = (
            ("Sustained projects", result.project_count, "project-ribbons"),
            ("Major themes", result.category_count, "theme-dynamics"),
            (
                "Cross-project connections",
                len(synthesis["cross_domain_connections"]),
                "connection-matrix",
            ),
            (
                "Unresolved open loops",
                sum(len(item["unresolved_work"]) for item in data["timelines"]),
                "open-work",
            ),
            (
                "Dormant threads resumed",
                len(synthesis["dormant_threads"]),
                "project-ribbons",
            ),
            (
                "Reversals or reframings",
                len(synthesis["tensions_and_reversals"]),
                "project-ribbons",
            ),
            (
                "Evidence-linked insights",
                len(synthesis["temporal_evolution"]),
                "semantic-network",
            ),
            (
                "Excluded or awaiting review",
                result.excluded_count + result.review_count,
                "exports",
            ),
        )
        atlas_results = html.Div(
            [
                html.Nav(
                    [
                        html.A(
                            [html.Strong(str(count)), html.Span(label)],
                            href=f"#{anchor}",
                            className="summary-card",
                        )
                        for label, count, anchor in summary_items
                    ],
                    className="summary-grid",
                    **_aria_props(aria_label="Recovered semantic structure"),
                ),
                dcc.Graph(
                    id="semantic-network",
                    figure=figures["network"],
                    config={"displaylogo": False, "responsive": True},
                ),
                dcc.Graph(
                    id="project-ribbons",
                    figure=figures["ribbons"],
                    config={"displaylogo": False, "responsive": True},
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="theme-dynamics",
                            figure=figures["dynamics"],
                            config={"displaylogo": False, "responsive": True},
                        ),
                        dcc.Graph(
                            id="connection-matrix",
                            figure=figures["matrix"],
                            config={"displaylogo": False, "responsive": True},
                        ),
                    ],
                    className="chart-grid",
                ),
                dcc.Graph(
                    id="open-work",
                    figure=figures["open_loops"],
                    config={"displaylogo": False, "responsive": True},
                ),
            ]
        )
        return progress, overview, exports, atlas_results, True

    @app.callback(
        Output("download-bundle", "data"),
        Input("download-button", "n_clicks"),
        State("job-id", "data"),
        prevent_initial_call=True,
    )
    def download(_clicks: int, job_id: str | None):  # type: ignore[no-untyped-def]
        job = jobs.get(job_id or "")
        if job is None or job.status is not JobStatus.SUCCEEDED:
            return no_update
        return make_download_payload(job.result.bundle_path)

    return app
