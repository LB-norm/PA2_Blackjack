import os
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

import pandas as pd
from pandas.io.formats.style import Styler


_ACTION_COLORS = {
    "hit": "#FF0000",        # red
    "stand": "#FFFF00",      # yellow
    "double": "#0000FF",     # blue
    "split": "#00B050",      # green
    "surrender": "#BFBFBF",  # grey
}


def _style_actions(df: pd.DataFrame) -> Styler:
    def fmt(v):
        s = str(v).lower()
        if s == "blackjack":
            return ""
        if s.startswith("double"):
            return f"background-color: {_ACTION_COLORS['double']}"
        if s.startswith("split"):
            return f"background-color: {_ACTION_COLORS['split']}"
        if "surrender" in s:
            return f"background-color: {_ACTION_COLORS['surrender']}"
        if s.startswith("stand") or s == "stay":
            return f"background-color: {_ACTION_COLORS['stand']}"
        if s.startswith("hit"):
            return f"background-color: {_ACTION_COLORS['hit']}"
        return ""

    return (
        df.style
        .applymap(fmt)
        .set_properties(**{"text-align": "center"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
    )


def _make_run_dir(base_dir: str, prefix: str = "run") -> str:
    """
    Creates base_dir/prefix_YYYYmmdd_HHMM (minute precision).
    If it already exists, appends _01, _02, ...
    """
    os.makedirs(base_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(base_dir, f"{prefix}_{stamp}")

    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=False)
        return run_dir

    i = 1
    while True:
        candidate = f"{run_dir}_{i:02d}"
        if not os.path.exists(candidate):
            os.makedirs(candidate, exist_ok=False)
            return candidate
        i += 1

def export_first_decision_agg(
    first_decision_agg: Dict[Tuple[Any, Any, Any], Dict[str, List[float]]],
    path: str = "first_decision_stats.xlsx",
) -> None:
    """
    first_decision_agg structure:
        {
            (cat, label, d_up): {
                action: [sum_return, count],
                ...
            },
            ...
        }

    Writes an Excel file with:
    - Sheet 'raw': one row per (state, action)
    - Sheet 'pivot_meanEV': one row per state, actions as columns with mean EV
    - Sheet 'pivot_counts': one row per state, actions as columns with counts
    """
    rows = []
    for (cat, label, d_up), action_dict in first_decision_agg.items():
        for action, (s, c) in action_dict.items():
            mean_ev = s / c if c > 0 else 0.0
            rows.append({
                "cat": cat,
                "label": label,
                "dealer_up": d_up,
                "action": action,
                "sum_return": s,
                "count": c,
                "mean_ev": mean_ev,
            })

    if not rows:
        return

    df = pd.DataFrame(rows)

    pivot_mean = df.pivot_table(
        index=["cat", "label", "dealer_up"],
        columns="action",
        values="mean_ev",
        aggfunc="first",
    )

    pivot_count = df.pivot_table(
        index=["cat", "label", "dealer_up"],
        columns="action",
        values="count",
        aggfunc="first",
    )

    # Ensure output directory exists
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Write to Excel with multiple sheets (openpyxl)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.sort_values(["cat", "label", "dealer_up", "action"]).to_excel(
            writer, sheet_name="raw", index=False
        )
        pivot_mean.sort_index().to_excel(writer, sheet_name="pivot_meanEV")
        pivot_count.sort_index().to_excel(writer, sheet_name="pivot_counts")

def export_results(save_dir: str, result: Dict[str, Any], up_cols: List[Any] | None = None) -> str:
    """
    Writes all outputs into a fresh run folder inside save_dir.
    Expects `result` to contain at least:
        - hard_grid, soft_grid, pair_grid
        - first_decision_agg
        - Q, N
        - rules (string or dict)

    Returns the created run folder path.
    """
    if up_cols is None:
        up_cols = [2, 3, 4, 5, 6, 7, 8, 9, 10, "A"]

    run_dir = _make_run_dir(save_dir, prefix="run")

    hard_grid = result["hard_grid"]
    soft_grid = result["soft_grid"]
    pair_grid = result["pair_grid"]
    first_decision_agg = result["first_decision_agg"]
    Q = result["Q"]
    N = result["N"]
    rules = result.get("rules", "")

    # --- rules.json ---
    rules_path = os.path.join(run_dir, "rules.json")
    with open(rules_path, "w", encoding="utf-8") as f:
        # Store as JSON in a robust way (even if rules is already a string)
        json.dump({"rules": rules}, f, indent=2, ensure_ascii=False, default=str)

    # --- grids (.xlsx) ---
    def write_grid_xlsx(name: str, grid: Dict[Any, Dict[Any, Any]], row_order: List[Any]) -> None:
        path = os.path.join(run_dir, f"{name}.xlsx")
        df = pd.DataFrame.from_dict(grid, orient="index").reindex(columns=up_cols)

        # order rows if possible
        df = df.reindex(row_order)

        df.index.name = "player"
        styler = _style_actions(df)

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            styler.to_excel(writer, sheet_name=name)

    write_grid_xlsx("hard", hard_grid, list(range(5, 22)))
    write_grid_xlsx("soft", soft_grid, list(range(13, 22)))
    write_grid_xlsx("pairs", pair_grid, [str(x) for x in [2, 3, 4, 5, 6, 7, 8, 9, 10, "A"]])

    # --- first_decision_ev.csv ---
    first_decision_path = os.path.join(run_dir, "first_decision_stats.xlsx")
    export_first_decision_agg(first_decision_agg, first_decision_path)
    return run_dir

