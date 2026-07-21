"""Executable compatibility module for the Dash application factory."""

from __future__ import annotations

from chatgpt_archive_compiler.app.factory import create_app

if __name__ == "__main__":
    create_app().run(debug=True)
