"""Dash application factory with isolated local state and thin callbacks."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from dash import Dash, Input, Output, State, dcc, html, no_update

from chatgpt_archive_compiler.app.jobs import JobRegistry, JobStatus
from chatgpt_archive_compiler.app.service import compile_for_app, preflight_archive
from chatgpt_archive_compiler.app.state import SessionStore
from chatgpt_archive_compiler.version import __version__

MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def _layout() -> html.Main:
    return html.Main(
        [
            dcc.Store(id="session-id"),
            dcc.Store(id="job-id"),
            dcc.Interval(id="job-poll", interval=750, disabled=True),
            html.Header(
                [
                    html.P("LOCAL-FIRST · EVIDENCE-LINKED · VERIFIABLE", className="eyebrow"),
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
                                options=[{"label": " Professional-safe mode", "value": "enabled"}],
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
                                "Compile Archive", id="compile", disabled=True, className="primary"
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
                            html.H2("Compilation progress"),
                            html.Div(
                                id="progress", children="Waiting for a safe archive preflight."
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
                        id="exports", children="Verified artifacts are available after compilation."
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
            return no_update, f"Preflight failed safely ({type(exception).__name__}).", True
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
        Output("job-poll", "disabled", allow_duplicate=True),
        Input("job-poll", "n_intervals"),
        State("job-id", "data"),
        prevent_initial_call=True,
    )
    def poll(_ticks: int, job_id: str | None):  # type: ignore[no-untyped-def]
        job = jobs.get(job_id or "")
        if job is None:
            return "Unknown local job.", no_update, no_update, True
        elapsed = (
            0
            if job.started_at is None
            else (job.finished_at or __import__("time").monotonic()) - job.started_at
        )
        progress = html.Div(
            [
                html.Strong(job.stage),
                html.P(f"{len(job.completed_stages)} stages completed · {elapsed:.1f}s elapsed"),
            ]
        )
        if job.status is JobStatus.FAILED:
            return job.error, no_update, no_update, True
        if job.status is not JobStatus.SUCCEEDED:
            return progress, no_update, no_update, False
        result = job.result
        overview = html.Div(
            [
                html.Div(
                    [html.Strong(str(result.project_count)), html.Span(" project threads")],
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
                    [html.Strong(str(result.included_count)), html.Span(" approved conversations")],
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
        return progress, overview, exports, True

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
        return dcc.send_file(job.result.bundle_path)

    return app
