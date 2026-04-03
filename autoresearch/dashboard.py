#!/usr/bin/env python3
"""
Autoresearch Dashboard — Live monitoring of experiment progress.

Usage:
    python dashboard.py                  # Auto-refresh every 30s
    python dashboard.py --once           # Print once and exit
    python dashboard.py --port 8050      # Launch web dashboard

Reads results.tsv and displays:
- Overall progress and best experiment
- Per-category accuracy trends
- Experiment timeline
- API token usage estimate
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_TSV = os.path.join(SCRIPT_DIR, "results.tsv")
TRAIN_PY = os.path.join(SCRIPT_DIR, "train.py")
STATUS_JSON = os.path.join(SCRIPT_DIR, "status.json")

CATEGORIES = [
    "NUMBER_SYSTEM", "UNIT_CONVERSION", "PHYSICS",
    "TEXT_ENCRYPTION", "BIT_MANIPULATION", "SYMBOL_TRANSFORM",
]

# Approximate token costs (USD per 1M tokens) for Claude Sonnet
COST_PER_1M_INPUT = 3.00
COST_PER_1M_OUTPUT = 15.00
# Rough estimate: ~4000 input tokens + ~3000 output tokens per experiment
EST_INPUT_TOKENS_PER_EXP = 4000
EST_OUTPUT_TOKENS_PER_EXP = 3000


def load_results():
    """Load results.tsv into list of dicts."""
    if not os.path.exists(RESULTS_TSV):
        return []
    rows = []
    with open(RESULTS_TSV, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    return rows


def load_status():
    """Load current run status from status.json."""
    if not os.path.exists(STATUS_JSON):
        return None
    try:
        with open(STATUS_JSON, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def bar(value, max_val=1.0, width=20):
    """Create a text progress bar."""
    if value is None or value == 'N/A':
        return ' ' * width + ' N/A'
    v = float(value)
    filled = int(v / max_val * width)
    filled = min(filled, width)
    return '\u2588' * filled + '\u2591' * (width - filled) + f' {v:.1%}'


def category_short(cat):
    """Shorten category name for display."""
    names = {
        "NUMBER_SYSTEM": "Number   ",
        "UNIT_CONVERSION": "Unit Conv",
        "PHYSICS": "Physics  ",
        "TEXT_ENCRYPTION": "Encrypt  ",
        "BIT_MANIPULATION": "Bit Manip",
        "SYMBOL_TRANSFORM": "Symbol   ",
    }
    return names.get(cat, cat[:9])


def sparkline(values, width=20):
    """Create a text sparkline from a list of floats."""
    if not values:
        return ''
    blocks = ' \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588'
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1
    # Sample values to fit width
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    return ''.join(blocks[min(int((v - mn) / rng * 7) + 1, 8)] for v in sampled)


def render_dashboard(rows):
    """Render the terminal dashboard."""
    lines = []
    w = 70

    lines.append('\033[2J\033[H')  # Clear screen
    lines.append('\033[1m' + '=' * w + '\033[0m')
    lines.append('\033[1m  AUTORESEARCH DASHBOARD\033[0m')
    lines.append('\033[1m' + '=' * w + '\033[0m')

    if not rows:
        lines.append('\n  No experiments yet. Waiting for results...\n')
        return '\n'.join(lines)

    # Filter successful experiments
    ok_rows = [r for r in rows if r.get('status') == 'OK']
    failed = [r for r in rows if r.get('status') != 'OK']
    total_exp = len(rows)

    lines.append(f'\n  Experiments: {total_exp} total | {len(ok_rows)} OK | {len(failed)} failed')

    # Best experiment
    if ok_rows:
        best = max(ok_rows, key=lambda r: float(r.get('overall_acc', 0)))
        best_acc = float(best['overall_acc'])
        lines.append(f'  \033[32mBest: Exp #{best["exp"]} — {best_acc:.1%} ({best["description"]})\033[0m')

        # Latest experiment
        latest = ok_rows[-1]
        latest_acc = float(latest['overall_acc'])
        trend = ''
        if len(ok_rows) >= 2:
            prev_acc = float(ok_rows[-2]['overall_acc'])
            diff = latest_acc - prev_acc
            if diff > 0:
                trend = f' \033[32m(+{diff:.1%})\033[0m'
            elif diff < 0:
                trend = f' \033[31m({diff:.1%})\033[0m'
            else:
                trend = ' (=)'
        lines.append(f'  Latest: Exp #{latest["exp"]} — {latest_acc:.1%}{trend} ({latest["description"]})')
    else:
        lines.append('  No successful experiments yet.')

    # Overall accuracy trend
    lines.append(f'\n  {"─" * w}')
    lines.append('  \033[1mOverall Accuracy Trend\033[0m')
    if ok_rows:
        accs = [float(r['overall_acc']) for r in ok_rows]
        lines.append(f'  {sparkline(accs, 50)}')
        lines.append(f'  Min: {min(accs):.1%}  Max: {max(accs):.1%}  Latest: {accs[-1]:.1%}')

    # Per-category breakdown (latest successful experiment)
    lines.append(f'\n  {"─" * w}')
    lines.append('  \033[1mPer-Category Accuracy (Latest Successful)\033[0m')
    if ok_rows:
        latest = ok_rows[-1]
        best_row = max(ok_rows, key=lambda r: float(r.get('overall_acc', 0)))
        lines.append(f'  {"Category":<12} {"Latest":>8} {"Best":>8}  {"Bar (Latest)":<25}')
        lines.append(f'  {"─" * 55}')
        for cat in CATEGORIES:
            latest_val = latest.get(cat, 'N/A')
            # Find best for this category
            best_cat = max(
                (float(r.get(cat, 0)) for r in ok_rows if r.get(cat, 'N/A') != 'N/A'),
                default=0
            )
            latest_display = f'{float(latest_val):.1%}' if latest_val != 'N/A' else 'N/A'
            best_display = f'{best_cat:.1%}'
            lines.append(f'  {category_short(cat):<12} {latest_display:>8} {best_display:>8}  {bar(latest_val)}')

    # Experiment history table
    lines.append(f'\n  {"─" * w}')
    lines.append('  \033[1mExperiment History (last 10)\033[0m')
    lines.append(f'  {"#":>3} {"Acc":>6} {"Loss":>7} {"Time":>6} {"St":>4}  {"Description":<35}')
    lines.append(f'  {"─" * 65}')
    for row in rows[-10:]:
        exp = row.get('exp', '?')
        acc = row.get('overall_acc', 'N/A')
        acc_str = f'{float(acc):.1%}' if acc != 'N/A' else ' N/A '
        loss = row.get('train_loss', 'N/A')
        loss_str = f'{float(loss):.4f}' if loss != 'N/A' else '  N/A '
        t = row.get('train_time_min', 'N/A')
        t_str = f'{float(t):.0f}m' if t != 'N/A' else ' N/A'
        status = row.get('status', '?')
        st_str = '\033[32mOK\033[0m  ' if status == 'OK' else f'\033[31m{status[:4]}\033[0m'
        desc = row.get('description', '')[:35]
        lines.append(f'  {exp:>3} {acc_str:>6} {loss_str:>7} {t_str:>6} {st_str}  {desc}')

    # Estimated API cost
    lines.append(f'\n  {"─" * w}')
    est_input = total_exp * EST_INPUT_TOKENS_PER_EXP
    est_output = total_exp * EST_OUTPUT_TOKENS_PER_EXP
    est_cost = (est_input / 1_000_000 * COST_PER_1M_INPUT +
                est_output / 1_000_000 * COST_PER_1M_OUTPUT)
    lines.append(f'  Est. API cost: ~${est_cost:.2f} ({total_exp} calls, ~{est_input/1000:.0f}K in / ~{est_output/1000:.0f}K out)')

    # Timestamp
    lines.append(f'\n  Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('  Press Ctrl+C to exit')
    lines.append('\033[1m' + '=' * w + '\033[0m')

    return '\n'.join(lines)


def run_terminal_dashboard(refresh_interval=30, once=False):
    """Run the terminal dashboard with auto-refresh."""
    try:
        while True:
            rows = load_results()
            output = render_dashboard(rows)
            print(output)
            if once:
                break
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print('\n  Dashboard stopped.')


def run_web_dashboard(port=8050):
    """Launch a web-based dashboard using plotly/dash."""
    try:
        import dash
        from dash import dcc, html, dash_table
        from dash.dependencies import Input, Output
        import plotly.graph_objects as go
    except ImportError:
        print("Web dashboard requires: pip install dash plotly")
        print("Falling back to terminal dashboard...")
        run_terminal_dashboard()
        return

    DARK_BG = '#0d1117'
    DARK_CARD = '#161b22'
    DARK_BORDER = '#30363d'
    DARK_TEXT = '#e6edf3'
    DARK_MUTED = '#8b949e'

    app = dash.Dash(__name__)

    app.index_string = '''
<!DOCTYPE html>
<html>
    <head>{%metas%}<title>Autoresearch</title>{%favicon%}{%css%}
    <style>
        body { background-color: ''' + DARK_BG + '''; color: ''' + DARK_TEXT + '''; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace; margin: 0; }
        * { box-sizing: border-box; }
        .Select-control, .Select-menu-outer { background-color: ''' + DARK_CARD + ''' !important; color: ''' + DARK_TEXT + ''' !important; border-color: ''' + DARK_BORDER + ''' !important; }
    </style>
    </head>
    <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>'''

    app.layout = html.Div([
        html.H1("Autoresearch Dashboard", style={
            'textAlign': 'center', 'color': DARK_TEXT, 'padding': '20px 0 10px',
            'borderBottom': f'1px solid {DARK_BORDER}', 'marginBottom': '20px',
        }),
        dcc.Interval(id='interval', interval=10_000, n_intervals=0),
        html.Div(id='live-status', style={
            'padding': '16px', 'margin': '10px 0', 'borderRadius': '8px',
            'border': f'2px solid {DARK_BORDER}', 'backgroundColor': DARK_CARD, 'fontSize': '15px',
        }),
        html.Div(id='summary', style={'padding': '10px 20px', 'fontSize': '16px', 'color': DARK_MUTED}),
        dcc.Graph(id='accuracy-trend'),
        dcc.Graph(id='category-bars'),
        html.H3("Experiment Log", style={'color': DARK_TEXT, 'padding': '0 0 8px'}),
        dash_table.DataTable(
            id='experiment-table',
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left', 'padding': '8px',
                'backgroundColor': DARK_CARD, 'color': DARK_TEXT,
                'border': f'1px solid {DARK_BORDER}',
            },
            style_header={
                'fontWeight': 'bold', 'backgroundColor': '#21262d',
                'color': DARK_TEXT, 'border': f'1px solid {DARK_BORDER}',
            },
            style_data_conditional=[
                {'if': {'filter_query': '{status} = OK'}, 'color': '#3fb950'},
                {'if': {'filter_query': '{status} = TRAIN_FAILED'}, 'color': '#f85149'},
                {'if': {'filter_query': '{status} = EVAL_FAILED'}, 'color': '#d29922'},
                {'if': {'filter_query': '{status} = ERROR'}, 'color': '#f85149'},
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#0d1117'},
            ],
            page_size=20,
        ),
    ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px', 'backgroundColor': DARK_BG})

    @app.callback(
        [Output('live-status', 'children'),
         Output('live-status', 'style'),
         Output('summary', 'children'),
         Output('accuracy-trend', 'figure'),
         Output('category-bars', 'figure'),
         Output('experiment-table', 'data'),
         Output('experiment-table', 'columns')],
        [Input('interval', 'n_intervals')]
    )
    def update(n):
        import traceback as _tb
        try:
            return _update(n)
        except Exception as _e:
            _tb.print_exc()
            empty_fig = go.Figure()
            return f"ERROR: {_e}", {'padding':'16px','color':'red'}, f"Callback error: {_e}", empty_fig, empty_fig, [], []

    def _update(n):
        rows = load_results()
        ok_rows = [r for r in rows if r.get('status') == 'OK']
        status = load_status()

        base_style = {
            'padding': '16px', 'margin': '10px 0', 'borderRadius': '8px',
            'border': '2px solid #30363d', 'fontSize': '15px',
        }
        try:
            if status:
                phase = status.get('phase', 'idle')
                phase_labels = {
                    'asking_agent': ('Asking Claude for next experiment...', '#2d2a1e', '#e3b341'),
                    'training':     ('Training model...', '#0d2137', '#58a6ff'),
                    'evaluating':   ('Evaluating model...', '#0d2119', '#3fb950'),
                    'idle':         ('Preparing next experiment...', '#161b22', '#8b949e'),
                }
                label, bg, fg = phase_labels.get(phase, ('Running...', '#161b22', '#e6edf3'))
                exp_num = status.get('exp_num', '?')
                max_exp = status.get('max_experiments', '?')
                desc = status.get('description', '')
                exp_started = status.get('exp_started_at', '')
                updated = status.get('updated_at', '')
                elapsed = ''
                if exp_started:
                    started_dt = datetime.strptime(exp_started, "%Y-%m-%d %H:%M:%S")
                    secs = int((datetime.now() - started_dt).total_seconds())
                    elapsed = f" | {secs//60}m {secs%60}s elapsed"
                plan = status.get('plan', '') or desc
                header = f"Experiment {exp_num}/{max_exp}{elapsed} — {label}"
                plan_line = f"Plan: {plan}" if plan else ""
                updated_line = f"Updated: {updated}"
                status_text = "\n".join(filter(None, [header, plan_line, updated_line]))
                live_style = {**base_style, 'backgroundColor': bg, 'color': fg, 'borderColor': fg, 'whiteSpace': 'pre-line'}
            else:
                status_text = "No active run. Start run_loop.py to begin."
                live_style = {**base_style, 'backgroundColor': '#161b22', 'color': '#8b949e'}
        except Exception as e:
            status_text = f"Status error: {e}"
            live_style = {**base_style, 'backgroundColor': '#fff0f0', 'color': '#900'}

        # Summary
        if ok_rows:
            best = max(ok_rows, key=lambda r: float(r.get('overall_acc', 0)))
            summary = f"Experiments: {len(rows)} total | {len(ok_rows)} OK | Best: #{best['exp']} at {float(best['overall_acc']):.1%} | Latest OK: #{ok_rows[-1]['exp']} at {float(ok_rows[-1]['overall_acc']):.1%}"
        else:
            summary = f"Experiments: {len(rows)} total | No successful runs yet"

        # Accuracy trend
        trend_fig = go.Figure()
        if ok_rows:
            exps = [int(r['exp']) for r in ok_rows]
            accs = [float(r['overall_acc']) for r in ok_rows]
            trend_fig.add_trace(go.Scatter(x=exps, y=accs, mode='lines+markers', name='Overall'))
            for cat in CATEGORIES:
                cat_accs = [float(r.get(cat, 0)) for r in ok_rows]
                trend_fig.add_trace(go.Scatter(x=exps, y=cat_accs, mode='lines', name=cat, opacity=0.5))
        trend_fig.update_layout(
            title='Accuracy Over Experiments', xaxis_title='Experiment #',
            yaxis_title='Accuracy', yaxis_range=[0, 1],
            template='plotly_dark', paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
            font_color='#e6edf3',
        )

        # Category bars (latest)
        cat_fig = go.Figure()
        if ok_rows:
            latest = ok_rows[-1]
            cats = CATEGORIES
            vals = [float(latest.get(c, 0)) for c in cats]
            colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in vals]
            cat_fig.add_trace(go.Bar(x=[category_short(c) for c in cats], y=vals,
                                     marker_color=colors))
        cat_fig.update_layout(
            title='Per-Category Accuracy (Latest)', yaxis_range=[0, 1],
            yaxis_title='Accuracy', template='plotly_dark',
            paper_bgcolor='#161b22', plot_bgcolor='#0d1117', font_color='#e6edf3',
        )

        # Table
        columns = [{'name': c, 'id': c} for c in ['exp', 'description', 'overall_acc', 'status', 'train_loss', 'train_time_min']]
        table_data = list(reversed(rows))

        return status_text, live_style, summary, trend_fig, cat_fig, table_data, columns

    print(f"Dashboard running at http://localhost:{port}")
    app.run(debug=False, port=port, host='0.0.0.0')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoresearch Dashboard")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    parser.add_argument("--refresh", type=int, default=30, help="Refresh interval in seconds")
    parser.add_argument("--web", action="store_true", help="Launch web dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Web dashboard port")
    args = parser.parse_args()

    if args.web:
        run_web_dashboard(args.port)
    else:
        run_terminal_dashboard(args.refresh, args.once)
