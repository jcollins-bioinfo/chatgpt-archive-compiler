"""Minimal Dash app shell.

The application layer should remain thin. Parser, redaction, and rendering logic
belongs in package modules that can also be called from CLI and tests.
"""

from __future__ import annotations

from dash import Dash, dcc, html


def create_app() -> Dash:
    """Create the Dash application."""
    app = Dash(__name__, title="ChatGPT Archive Compiler")

    app.layout = html.Main(
        [
            html.H1("ChatGPT Archive Compiler"),
            html.P(
                "Local-first compiler for exported conversation archives. "
                "Upload and parsing workflows will be implemented after the Archive IR stabilizes."
            ),
            dcc.Upload(
                id="upload-export-zip",
                children=html.Div(["Drag and drop or select an export ZIP"]),
                multiple=False,
            ),
            html.Div(id="upload-status"),
        ],
        style={"maxWidth": "960px", "margin": "48px auto", "fontFamily": "system-ui"},
    )
    return app


if __name__ == "__main__":
    create_app().run(debug=True)
