#!/usr/bin/env python3
"""Simple autoresearch dashboard."""
import csv
import json
import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_TSV = os.path.join(SCRIPT_DIR, "results.tsv")
STATUS_JSON = os.path.join(SCRIPT_DIR, "status.json")
CATEGORIES = ["NUMBER_SYSTEM", "UNIT_CONVERSION", "PHYSICS",
               "TEXT_ENCRYPTION", "BIT_MANIPULATION", "SYMBOL_TRANSFORM"]


def load_results():
    if not os.path.exists(RESULTS_TSV):
        return []
    with open(RESULTS_TSV, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_status():
    if not os.path.exists(STATUS_JSON):
        return None
    try:
        with open(STATUS_JSON, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def run(port=8050):
    import dash
    from dash import dcc, html, dash_table
    import plotly.graph_objects as go

    BG       = "#0f1117"
    SURFACE  = "#1a1d27"
    BORDER   = "#2e3347"
    TEXT     = "#e2e8f0"
    MUTED    = "#8892a4"
    ACCENT   = "#5b8dee"

    PLOT_THEME = dict(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=TEXT, size=12),
    )
    AXIS_STYLE = dict(gridcolor=BORDER, zerolinecolor=BORDER)

    app = dash.Dash(__name__, suppress_callback_exceptions=True,
                    external_stylesheets=[],
                    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
    app.index_string = '''<!DOCTYPE html>
<html>
<head>{%metas%}<title>Autoresearch</title>{%favicon%}{%css%}
<style>
  body, html { background: #0f1117 !important; margin: 0; padding: 0; }
  * { box-sizing: border-box; }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>'''

    app.layout = html.Div(style={
        "maxWidth": "1200px", "margin": "0 auto", "padding": "20px",
        "fontFamily": "'Inter', 'Segoe UI', sans-serif",
        "backgroundColor": BG, "minHeight": "100vh", "color": TEXT,
    }, children=[
        html.H2("Autoresearch Dashboard", style={"color": TEXT, "marginBottom": "4px"}),
        dcc.Interval(id="tick", interval=10_000, n_intervals=0),

        # Live status
        html.Div(id="status-box", style={"padding": "12px", "borderRadius": "8px",
                                          "border": f"1px solid {BORDER}", "marginBottom": "16px"}),

        # Summary line
        html.Div(id="summary-line", style={"marginBottom": "16px", "fontSize": "15px", "color": MUTED}),

        # Charts
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(dcc.Graph(id="trend-chart"), style={"flex": "1"}),
            html.Div(dcc.Graph(id="cat-chart"),   style={"flex": "1"}),
        ]),

        # Table toggle
        html.Div(style={"display": "flex", "gap": "8px", "marginTop": "24px", "marginBottom": "8px",
                        "alignItems": "center"}, children=[
            html.H3("Experiment Log", style={"color": TEXT, "margin": "0"}),
            html.Button("Show All", id="toggle-btn", n_clicks=0,
                        style={"padding": "4px 12px", "fontSize": "12px", "cursor": "pointer",
                               "backgroundColor": SURFACE, "color": MUTED,
                               "border": f"1px solid {BORDER}", "borderRadius": "6px"}),
        ]),
        dash_table.DataTable(
            id="exp-table",
            style_table={"overflowX": "auto", "borderRadius": "8px", "border": f"1px solid {BORDER}"},
            style_cell={
                "textAlign": "left", "padding": "8px 10px", "fontSize": "13px",
                "backgroundColor": SURFACE, "color": TEXT, "border": f"1px solid {BORDER}",
                "whiteSpace": "normal", "height": "auto", "lineHeight": "1.5",
                "verticalAlign": "top",
            },
            style_cell_conditional=[
                {"if": {"column_id": "description"}, "minWidth": "340px", "maxWidth": "520px",
                 "whiteSpace": "normal", "wordBreak": "break-word"},
                {"if": {"column_id": "exp"},           "width": "48px",  "textAlign": "center"},
                {"if": {"column_id": "timestamp"},     "width": "110px"},
                {"if": {"column_id": "overall_acc"},   "width": "80px",  "textAlign": "right"},
                {"if": {"column_id": "status"},        "width": "100px"},
                {"if": {"column_id": "train_loss"},    "width": "76px",  "textAlign": "right"},
                {"if": {"column_id": "train_time_min"},"width": "72px",  "textAlign": "right"},
            ],
            style_header={"fontWeight": "bold", "backgroundColor": "#252836",
                          "color": TEXT, "border": f"1px solid {BORDER}"},
            style_data_conditional=[
                {"if": {"filter_query": '{status} = "OK"'},          "color": "#4ade80"},
                {"if": {"filter_query": '{status} = "EVAL_FAILED"'}, "color": "#fb923c"},
                {"if": {"column_id": "description"},                 "color": MUTED},
            ],
            page_size=25,
        ),
    ])

    @app.callback(
        [dash.Output("status-box", "children"),
         dash.Output("status-box", "style"),
         dash.Output("summary-line", "children"),
         dash.Output("trend-chart", "figure"),
         dash.Output("cat-chart", "figure"),
         dash.Output("exp-table", "data"),
         dash.Output("exp-table", "columns"),
         dash.Output("toggle-btn", "children")],
        [dash.Input("tick", "n_intervals"),
         dash.Input("toggle-btn", "n_clicks")],
    )
    def refresh(n, n_clicks):
        rows = load_results()
        ok = [r for r in rows if r.get("status") == "OK"]
        failed = [r for r in rows if r.get("status") in ("TRAIN_FAILED", "EVAL_FAILED", "ERROR")]
        status = load_status()

        show_all = bool(n_clicks and n_clicks % 2 == 1)
        toggle_label = "Show OK Only" if show_all else "Show All"

        # --- Status box ---
        COLORS = {
            "asking_agent": ("#2d2a14", "#fde68a", "Asking Claude for next idea..."),
            "training":     ("#0d1f3c", "#93c5fd", "Training model..."),
            "evaluating":   ("#0d2818", "#86efac", "Evaluating model..."),
            "idle":         ("#1a1d27", "#94a3b8", "Idle / preparing next..."),
        }
        base = {"padding": "12px", "borderRadius": "8px", "border": "1px solid #2e3347", "marginBottom": "16px"}
        if status:
            phase = status.get("phase", "idle")
            bg, fg, label = COLORS.get(phase, ("#1a1d27", "#e2e8f0", phase))
            exp_n = status.get("exp_num", "?")
            max_n = status.get("max_experiments", "?")
            desc  = status.get("plan", "") or status.get("description", "")
            upd   = status.get("updated_at", "")
            elapsed = ""
            try:
                t0 = datetime.strptime(status.get("exp_started_at", ""), "%Y-%m-%d %H:%M:%S")
                s = int((datetime.now() - t0).total_seconds())
                elapsed = f" | {s//60}m {s%60}s elapsed"
            except Exception:
                pass
            box_header = f"Exp {exp_n}/{max_n}{elapsed}  —  {label}  |  Updated: {upd}"
            box_plan   = desc if desc else "(no plan yet)"
            box_children = [
                html.Div(box_header, style={"fontWeight": "600", "marginBottom": "6px"}),
                html.Div([html.Span("Plan: ", style={"opacity": "0.6"}), box_plan],
                         style={"whiteSpace": "pre-wrap", "wordBreak": "break-word",
                                "fontSize": "14px", "lineHeight": "1.6"}),
            ]
            box_text  = box_children  # will be used as children below
            box_style = {**base, "backgroundColor": bg, "color": fg, "borderColor": fg}
        else:
            box_text  = [html.Div("No active run. Launch run_loop.py to start.")]
            box_style = {**base, "backgroundColor": "#1a1d27", "color": "#8892a4"}

        # --- Summary ---
        if ok:
            best = max(ok, key=lambda r: float(r.get("overall_acc", 0)))
            summ = (f"{len(rows)} total  |  {len(ok)} OK  |  {len(failed)} failed  |  "
                    f"Best: #{best['exp']} = {float(best['overall_acc']):.1%}  |  "
                    f"Latest OK: #{ok[-1]['exp']} = {float(ok[-1]['overall_acc']):.1%}")
        else:
            summ = f"{len(rows)} total  |  {len(ok)} OK  |  {len(failed)} failed  |  No successful runs yet"

        # --- Trend chart ---
        trend = go.Figure()
        if ok:
            xs = [int(r["exp"]) for r in ok]
            trend.add_trace(go.Scatter(x=xs, y=[float(r["overall_acc"]) for r in ok],
                                       mode="lines+markers", name="Overall", line=dict(width=3)))
            for cat in CATEGORIES:
                trend.add_trace(go.Scatter(x=xs, y=[float(r.get(cat, 0)) for r in ok],
                                           mode="lines", name=cat, opacity=0.45))
        trend.update_layout(title="Accuracy Over Experiments", xaxis_title="Exp #",
                            xaxis={**AXIS_STYLE}, yaxis={**AXIS_STYLE, "range": [0, 1], "title": "Accuracy"},
                            margin=dict(l=40, r=10, t=40, b=30), **PLOT_THEME)

        # --- Category chart ---
        cats = go.Figure()
        if ok:
            vals = [float(ok[-1].get(c, 0)) for c in CATEGORIES]
            cats.add_trace(go.Bar(
                x=[c.replace("_", " ").title() for c in CATEGORIES], y=vals,
                marker_color=["#4ade80" if v > 0.7 else "#fb923c" if v > 0.4 else "#f87171" for v in vals],
            ))
        cats.update_layout(title="Per-Category (Latest OK)",
                           xaxis={**AXIS_STYLE}, yaxis={**AXIS_STYLE, "range": [0, 1], "title": "Accuracy"},
                           margin=dict(l=40, r=10, t=40, b=30), **PLOT_THEME)

        # --- Table: default shows only OK + EVAL_FAILED, toggle shows all ---
        table_rows = rows if show_all else [r for r in rows if r.get("status") not in ("TRAIN_FAILED", "ERROR")]
        cols = [{"name": c, "id": c} for c in
                ["exp", "timestamp", "description", "overall_acc", "status", "train_loss", "train_time_min"]]
        return box_text, box_style, summ, trend, cats, list(reversed(table_rows)), cols, toggle_label

    print(f"Dashboard at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    run(port)
