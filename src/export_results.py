import os
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

import pandas as pd
from pandas.io.formats.style import Styler

Action = str

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

    Q = result["Q"]
    N = result["N"]
    rules = result["rules"]

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
    export_initial_decision_qn_tables(save_dir, Q, N, rules, up_cols=up_cols)
    return run_dir

def export_initial_decision_qn_tables(
    save_path: str,
    Q: Dict[Tuple, Dict[Action, float]],
    N: Dict[Tuple, Dict[Action, int]],
    rules: Any,
    *,
    up_cols,
    hard_totals=range(5, 22),   # 5..21
    soft_totals=range(13, 22),  # 13..21
    pair_rows = None,  # [2..10,'A']
    filename = None,
) -> str:
    """
    Exports initial-decision diagnostics from Q and N:
      - how often each action was taken (N)
      - estimated EV for each action (Q)
    for the *initial hand* abstraction used by your strategy grids.

    Writes one Excel file with multiple sheets:
      - all_long:   long format rows (state, action)
      - hard_long, soft_long, pair_long: category subsets
      - all_wide:   one row per state with EV_* and N_* columns
      - state_best: one row per state with best_action/best_ev/visits

    Returns the written filepath.
    """
    os.makedirs(save_path, exist_ok=True)

    if up_cols is None:
        up_cols = [2, 3, 4, 5, 6, 7, 8, 9, 10, "A"]
    if pair_rows is None:
        pair_rows = [2, 3, 4, 5, 6, 7, 8, 9, 10, "A"]

    TEN_RANKS = {10, "T", "J", "Q", "K"}

    def canon_rank(x: Any) -> str:
        return "10" if x in TEN_RANKS else str(x)

    def canon_pair_rank(pr: Any) -> str:
        if pr is None:
            return None
        return "10" if pr in TEN_RANKS else str(pr)

    def double_range(rules_: Any) -> tuple[int, int]:
        da = str(rules_.double_allowed)
        if da == "any_two":
            return (4, 21)
        if da == "10-11":
            return (10, 11)
        if da == "9-11":
            return (9, 11)
        raise ValueError(f"unknown double_allowed={rules_.double_allowed!r}")

    def can_double_now(total: int, num_cards: int, after_split: bool, rules_: Any) -> bool:

        if num_cards != 2:
            return False
        if after_split and not bool(rules_.double_after_split):
            return False
        lo, hi = double_range(rules_)
        return lo <= total <= hi

    def encode_state(
        pl_total: int,
        pl_usable_ace: bool,
        d_up: Any,
        pr: Any,
        num_cards: int,
        after_split: bool,
        splits_done: int,
    ) -> Tuple:
        can_d = can_double_now(pl_total, num_cards, after_split, rules)
        can_spl = (
            bool(rules.allow_splits)
            and (num_cards == 2)
            and (pr is not None)
            and (splits_done < int(rules.max_splits))
        )
        return (
            pl_total,
            int(pl_usable_ace),
            canon_rank(d_up),
            canon_pair_rank(pr) if pr is not None else "0",
            num_cards,
            int(after_split),
            splits_done,
            int(can_d),
            int(can_spl),
        )

    def allowed_actions(state_key: Tuple, *, initial_hand: bool) -> List[Action]:
        (_, _, _, _, num_cards, after_split_i, _, can_d_i, can_spl_i) = state_key
        after_split = bool(after_split_i)
        can_d = bool(can_d_i)
        can_spl = bool(can_spl_i)

        acts: List[Action] = ["hit", "stand"]
        if can_d:
            acts.append("double")
        if can_spl:
            acts.append("split")

        # surrender only on original first decision, two cards, not after split
        if str(rules.allow_surrender).lower() != "none" and initial_hand and (not after_split) and num_cards == 2:
            acts.append("surrender")
        return acts

    # same tie-break you use later
    pref = {"split": 4, "double": 3, "surrender": 2, "stand": 1, "hit": 0}

    def pair_total_usable(pr: Any) -> tuple[int, bool]:
        if str(pr) == "A":
            return 12, True  # A,A
        v = 10 if (pr in TEN_RANKS or str(pr) == "10") else int(pr)
        return 2 * v, False

    rows_long: List[dict] = []

    def add_state_rows(cat: str, label: Any, up: Any, *, total: int, usable: bool, pr: Any):
        sk = encode_state(
            pl_total=total,
            pl_usable_ace=usable,
            d_up=up,
            pr=pr,
            num_cards=2,
            after_split=False,
            splits_done=0,
        )
        acts = allowed_actions(sk, initial_hand=True)

        qsa = Q.get(sk, {})
        nsa = N.get(sk, {})
        visits_total = int(sum(nsa.values()))

        # best among actions with known Q-values
        known_acts = [a for a in acts if a in qsa]
        if known_acts:
            best_a = max(known_acts, key=lambda a: (qsa.get(a, float("-inf")), pref.get(a, -1)))
            best_ev = float(qsa[best_a])
        else:
            best_a = "unknown"
            best_ev = float("nan")

        for a in acts:
            ev = qsa.get(a, float("nan"))
            cnt = int(nsa.get(a, 0))
            rows_long.append(
                dict(
                    category=cat,
                    label=str(label),
                    dealer_up=canon_rank(up),
                    action=a,
                    ev=ev,
                    count=cnt,
                    visits_total=visits_total,
                    best_action=best_a,
                    best_ev=best_ev,
                    state_key=str(sk),
                )
            )

    # hard
    for t in hard_totals:
        for up in up_cols:
            add_state_rows("hard", t, up, total=int(t), usable=False, pr=None)

    # soft
    for t in soft_totals:
        for up in up_cols:
            add_state_rows("soft", t, up, total=int(t), usable=True, pr=None)

    # pairs
    for pr0 in pair_rows:
        total, usable = pair_total_usable(pr0)
        for up in up_cols:
            add_state_rows("pair", pr0, up, total=total, usable=usable, pr=pr0)

    df_long = pd.DataFrame(rows_long)

    # Wide: one row per (category,label,dealer_up) with EV_* and N_* columns
    ev_wide = (
        df_long.pivot_table(
            index=["category", "label", "dealer_up", "state_key", "visits_total", "best_action", "best_ev"],
            columns="action",
            values="ev",
            aggfunc="first",
        )
        .add_prefix("EV_")
        .reset_index()
    )
    n_wide = (
        df_long.pivot_table(
            index=["category", "label", "dealer_up", "state_key"],
            columns="action",
            values="count",
            aggfunc="first",
        )
        .add_prefix("N_")
        .reset_index()
    )
    df_wide = ev_wide.merge(n_wide, on=["category", "label", "dealer_up", "state_key"], how="left")

    df_best = (
        df_wide[["category", "label", "dealer_up", "visits_total", "best_action", "best_ev", "state_key"]]
        .drop_duplicates()
        .sort_values(["category", "label", "dealer_up"])
        .reset_index(drop=True)
    )

    df_hard_long = df_long[df_long["category"] == "hard"].copy()
    df_soft_long = df_long[df_long["category"] == "soft"].copy()
    df_pair_long = df_long[df_long["category"] == "pair"].copy()

    if filename is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"initial_decision_QN_{stamp}.xlsx"
    out_path = os.path.join(save_path, filename)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_long.to_excel(writer, sheet_name="all_long", index=False)
        df_hard_long.to_excel(writer, sheet_name="hard_long", index=False)
        df_soft_long.to_excel(writer, sheet_name="soft_long", index=False)
        df_pair_long.to_excel(writer, sheet_name="pair_long", index=False)
        df_wide.to_excel(writer, sheet_name="all_wide", index=False)
        df_best.to_excel(writer, sheet_name="state_best", index=False)

        # light formatting: freeze header row + autofilter
        for name in ["all_long", "hard_long", "soft_long", "pair_long", "all_wide", "state_best"]:
            ws = writer.book[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

    return out_path
