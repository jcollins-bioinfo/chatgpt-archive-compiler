"""Privacy-aware Plotly figure specifications for the longitudinal atlas."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

Figure = dict[str, Any]
_COLORS = ("#73e0bf", "#a99aff", "#f4bd6a", "#6eb5ff", "#ff8ea1", "#9bd36a", "#d6a4ff")


def load_dashboard_data(path: Path) -> dict[str, Any]:
    """Load the local source-derived dashboard dossier."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Dashboard data must be an object.")
    return value


def _layout(title: str, *, height: int = 420) -> dict[str, Any]:
    """Return one WCAG-oriented dark Plotly layout shared by every chart."""

    return {
        "title": {"text": title, "font": {"color": "#eef2ff", "size": 18}},
        "height": height,
        "paper_bgcolor": "#111827",
        "plot_bgcolor": "#111827",
        "font": {"color": "#dbe4f5"},
        "hoverlabel": {"bgcolor": "#eef2ff", "font": {"color": "#111827"}},
        "legend": {"font": {"color": "#dbe4f5"}},
        "xaxis": {"gridcolor": "#33415c", "linecolor": "#71809f", "tickfont": {"color": "#c8d2e5"}},
        "yaxis": {"gridcolor": "#33415c", "linecolor": "#71809f", "tickfont": {"color": "#c8d2e5"}},
        "margin": {"l": 60, "r": 30, "t": 58, "b": 58},
        "clickmode": "event+select",
    }


def _aliases(projects: list[str], anonymous: bool, synthetic: bool) -> dict[str, str]:
    return {
        project: (project if not anonymous or synthetic else f"Project {chr(65 + index)}")
        for index, project in enumerate(projects)
    }


def build_figures(
    data: dict[str, Any], *, anonymous: bool, synthetic: bool = False
) -> dict[str, Figure]:
    """Build coordinated timeline, ribbon, dynamics, matrix, and open-work figures."""

    conversations = list(data.get("conversations", []))
    project_names = sorted(
        {project for item in conversations for project in item.get("projects", [])}
    )
    aliases = _aliases(project_names, anonymous, synthetic)
    lane = {project: index for index, project in enumerate(project_names)}
    by_key = {item["key"]: item for item in conversations}

    edge_x: list[object] = []
    edge_y: list[object] = []
    for edge in data.get("edges", []):
        left, right = by_key.get(edge["source_key"]), by_key.get(edge["target_key"])
        if left is None or right is None:
            continue
        left_project = next(iter(left.get("projects", [])), "")
        right_project = next(iter(right.get("projects", [])), "")
        if not left_project or not right_project or left_project != right_project:
            continue
        edge_x.extend((left.get("date"), right.get("date"), None))
        edge_y.extend((lane[left_project], lane[right_project], None))
    network_traces: list[dict[str, Any]] = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": edge_x,
            "y": edge_y,
            "line": {"color": "rgba(115,224,191,.22)", "width": 1},
            "hoverinfo": "skip",
            "showlegend": False,
        }
    ]
    for index, project in enumerate(project_names):
        members = [item for item in conversations if project in item.get("projects", [])]
        network_traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": aliases[project],
                "x": [item.get("date") for item in members],
                "y": [lane[project]] * len(members),
                "customdata": [item["key"] for item in members],
                "text": [
                    (
                        "Conversation " + str(conversations.index(item) + 1).zfill(3)
                        if anonymous and not synthetic
                        else item["title"]
                    )
                    + f"<br>{item.get('role')}<br>{item.get('evidence_count', 0)} connections"
                    for item in members
                ],
                "hovertemplate": "%{text}<br>%{x}<extra></extra>",
                "marker": {
                    "size": [
                        max(
                            8,
                            min(
                                22, 7 + item.get("message_count", 0) + item.get("evidence_count", 0)
                            ),
                        )
                        for item in members
                    ],
                    "color": _COLORS[index % len(_COLORS)],
                    "line": {"color": "#111827", "width": 1.5},
                },
            }
        )
    network_layout = _layout("Longitudinal semantic network", height=500)
    network_layout["yaxis"].update(
        {
            "tickvals": list(range(len(project_names))),
            "ticktext": [aliases[p] for p in project_names],
        }
    )

    ribbon_traces: list[dict[str, Any]] = []
    for index, timeline in enumerate(data.get("timelines", [])):
        name = timeline["project_name"]
        events = timeline.get("events", [])
        ribbon_traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": aliases.get(name, name),
                "x": [event.get("occurred_at") for event in events],
                "y": [index] * len(events),
                "text": [
                    event.get("label") if not anonymous or synthetic else f"Milestone {number + 1}"
                    for number, event in enumerate(events)
                ],
                "customdata": [event.get("conversation_keys", []) for event in events],
                "hovertemplate": "%{text}<br>%{x}<extra></extra>",
                "line": {"color": _COLORS[index % len(_COLORS)], "width": 7},
                "marker": {
                    "size": 11,
                    "color": "#eef2ff",
                    "line": {"color": _COLORS[index % len(_COLORS)], "width": 3},
                },
            }
        )
    ribbon_layout = _layout("Project evolution ribbons")
    ribbon_layout["yaxis"].update(
        {
            "tickvals": list(range(len(ribbon_traces))),
            "ticktext": [trace["name"] for trace in ribbon_traces],
        }
    )

    monthly: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for item in conversations:
        if item.get("date"):
            monthly[item["date"][:7]][item.get("theme", "Other")] += 1
    months = sorted(monthly)
    themes = [
        name
        for name, _count in Counter(
            item.get("theme", "Other") for item in conversations
        ).most_common(8)
    ]
    dynamics = [
        {
            "type": "scatter",
            "mode": "lines+markers",
            "stackgroup": "one",
            "name": theme,
            "x": months,
            "y": [monthly[month][theme] for month in months],
            "hovertemplate": "%{x}: %{y} conversations<extra>%{fullData.name}</extra>",
        }
        for theme in themes
    ]

    matrix = [[0 for _ in project_names] for _ in project_names]
    for item in conversations:
        members = [project for project in item.get("projects", []) if project in lane]
        for left in members:
            for right in members:
                matrix[lane[left]][lane[right]] += 1
    matrix_trace = {
        "type": "heatmap",
        "z": matrix,
        "x": [aliases[p] for p in project_names],
        "y": [aliases[p] for p in project_names],
        "colorscale": [[0, "#111827"], [0.45, "#315f78"], [1, "#73e0bf"]],
        "hovertemplate": "%{x} ↔ %{y}<br>%{z} shared conversations<extra></extra>",
    }

    open_rows: list[tuple[str, str, int]] = []
    for timeline in data.get("timelines", []):
        for number, question in enumerate(timeline.get("unresolved_work", [])):
            open_rows.append((timeline["project_name"], question, number))
    open_trace = {
        "type": "scatter",
        "mode": "markers",
        "x": [row[2] + 1 for row in open_rows],
        "y": [lane.get(row[0], 0) for row in open_rows],
        "text": [
            "Hidden in anonymous mode" if anonymous and not synthetic else row[1]
            for row in open_rows
        ],
        "hovertemplate": "%{text}<extra></extra>",
        "marker": {
            "size": [12 + min(12, row[2] * 2) for row in open_rows],
            "color": "#f4bd6a",
            "symbol": "diamond",
            "line": {"color": "#111827", "width": 1},
        },
    }
    open_layout = _layout("Unresolved-work landscape")
    open_layout["xaxis"].update({"title": "Recurrence within project (heuristic, not urgency)"})
    open_layout["yaxis"].update(
        {
            "tickvals": list(range(len(project_names))),
            "ticktext": [aliases[p] for p in project_names],
        }
    )

    return {
        "network": {"data": network_traces, "layout": network_layout},
        "ribbons": {"data": ribbon_traces, "layout": ribbon_layout},
        "dynamics": {"data": dynamics, "layout": _layout("Theme dynamics by month")},
        "matrix": {"data": [matrix_trace], "layout": _layout("Cross-project evidence matrix")},
        "open_loops": {"data": [open_trace], "layout": open_layout},
    }
