import pandas as pd
import os
from typing import Dict, Tuple, List, Optional, Any
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
            return ""  # no color
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

def export_results(save_dir, rules, results, up_cols):
    hard_grid = results["hard_grid"]
    soft_grid = results["soft_grid"]
    pair_grid = results["pair_grid"]
    first_decision_agg = results["first_decision_agg"]
    Q = results["Q"]
    N = results["N"]

    os.makedirs(save_dir, exist_ok=True)
    # Save params
    with open(os.path.join(save_dir, "rules.json"), "w") as f:
        f.write(str(rules))

    # Save grids as xlsx
    def write_grid_csv(name: str, grid: Dict, rows: List[Any]):
        path = os.path.join(save_dir, f"{name}.xlsx")
        df = pd.DataFrame.from_dict(grid, orient="index")[up_cols]
        df.index.name = "player"
        styler = _style_actions(df)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            styler.to_excel(writer, sheet_name=name)

    write_grid_csv("hard", hard_grid, list(range(5,22)))
    write_grid_csv("soft", soft_grid, list(range(13,22)))
    write_grid_csv("pairs", pair_grid, [2,3,4,5,6,7,8,9,10,'A'])

    # Save first-decision EVs for visualization
    with open(os.path.join(save_dir, "first_decision_ev.csv"), "w") as f:
        f.write("category,player,dealer_up,action,mean_ev,count\n")
        for (cat, lbl, up), acts in first_decision_agg.items():
            for a, (s, c) in acts.items():
                mean_ev = s / max(1, c)
                f.write(f"{cat},{lbl},{up},{a},{mean_ev:.6f},{c}\n")

    # Save a compact Q dump (only initial two-card states) for deeper analysis
    with open(os.path.join(save_dir, "q_initial_states.csv"), "w") as f:
        f.write("pl_total,usable_ace,dealer_up,pair_rank,num_cards,after_split,splits_done,split_aces,can_double,can_split,action,q,n\n")
        for sk, row in Q.items():
            (pl_total, usable_ace, d_up_s, pr_s, num_cards, after_split, splits_done, split_aces, can_d, can_spl) = sk
            if num_cards != 2 or after_split:
                continue
            for a, qv in row.items():
                n = N.get(sk, {}).get(a, 0)
                f.write(f"{pl_total},{usable_ace},{d_up_s},{pr_s},{num_cards},{after_split},{splits_done},{split_aces},{can_d},{can_spl},{a},{qv:.6f},{n}\n")

def export_first_decision_agg(first_decision_agg: Dict[Tuple[Any, Any, Any], Dict[str, List[float]]],
                              path: str = "first_decision_stats.xlsx") -> None:
    """
    first_decision_agg structure:
        {
            (cat, label, d_up): {
                action: [sum_return, count],
                ...
            },
            ...
        }

    This writes an Excel file with:
    - Sheet 'raw': one row per (state, action)
    - Sheet 'pivot_meanEV': one row per state, actions as columns with mean EV
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
        # nothing to export
        return

    df = pd.DataFrame(rows)

    # Pivot view: best for scanning strategy logic
    pivot_mean = df.pivot_table(
        index=["cat", "label", "dealer_up"],
        columns="action",
        values="mean_ev",
        aggfunc="first"
    )

    # Optional: also show sample counts per action
    pivot_count = df.pivot_table(
        index=["cat", "label", "dealer_up"],
        columns="action",
        values="count",
        aggfunc="first"
    )

    # Write to Excel with multiple sheets
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.sort_values(["cat", "label", "dealer_up", "action"]).to_excel(
            writer,
            sheet_name="raw",
            index=False
        )

        pivot_mean.sort_index().to_excel(writer, sheet_name="pivot_meanEV")
        pivot_count.sort_index().to_excel(writer, sheet_name="pivot_counts")

        # Optional formatting: autofit columns
        for sheet_name, dataf in {
            "raw": df,
            "pivot_meanEV": pivot_mean.reset_index(),
            "pivot_counts": pivot_count.reset_index(),
        }.items():
            worksheet = writer.sheets[sheet_name]
            for i, col in enumerate(dataf.columns):
                max_len = max(
                    len(str(col)),
                    *(len(str(v)) for v in dataf[col].astype(str))
                )
                worksheet.set_column(i, i, max_len + 2)