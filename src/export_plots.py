from __future__ import annotations
import math
from typing import Optional, Sequence
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

def _nice_log_ticks(x_min: float, x_max: float) -> list[int]:
    """1-2-5 log ticks between x_min and x_max (inclusive)."""
    if x_min <= 0 or x_max <= 0:
        return []
    p_min = int(math.floor(math.log10(x_min)))
    p_max = int(math.ceil(math.log10(x_max)))
    ticks: list[int] = []
    for p in range(p_min, p_max + 1):
        for m in (1, 2, 5):
            v = int(m * (10 ** p))
            if x_min <= v <= x_max:
                ticks.append(v)
    return sorted(set(ticks))


def _fmt_si(n: float) -> str:
    """Format numbers like 10000 -> 10k, 1000000 -> 1M."""
    n = float(n)
    if n >= 1e9:
        v = n / 1e9
        return f"{v:g}B" if v % 1 else f"{int(v)}B"
    if n >= 1e6:
        v = n / 1e6
        return f"{v:g}M" if v % 1 else f"{int(v)}M"
    if n >= 1e3:
        v = n / 1e3
        return f"{v:g}k" if v % 1 else f"{int(v)}k"
    return str(int(n))


def plot_eval_return(
    eval_history_xlsx_path: str,
    *,
    title: str = "Greedy mean Return vs Episode",
    show_ci: bool = True,
    ci_z: float = 1.96,  # 95% CI if stderr_return is present
) -> go.Figure:
    """
    Reads eval_history from an xlsx file and plots mean_return (%) over train_episode.
    Uses a log-scaled x-axis with 1-2-5 ticks to match log-spaced checkpoints.
    """
    df = pd.read_excel(eval_history_xlsx_path).copy()
    if "train_episode" not in df or "mean_return" not in df:
        raise ValueError("Expected columns: 'train_episode' and 'mean_return'.")

    df = df.sort_values("train_episode")
    x = df["train_episode"].astype(float).to_numpy()
    y = (df["mean_return"].astype(float) * 100.0).to_numpy()  # percent

    tickvals = _nice_log_ticks(float(x.min()), float(x.max()))
    ticktext = [_fmt_si(v) for v in tickvals]

    fig = go.Figure()

    # Optional CI band (if stderr_return exists)
    if show_ci and "stderr_return" in df:
        se = df["stderr_return"].astype(float).to_numpy() * 100.0
        ci = ci_z * se

        fig.add_trace(go.Scatter(
            x=x, y=y + ci,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="CI upper",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y - ci,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name=f"95% CI",
            hoverinfo="skip",
            fillcolor="rgba(255,0,0,0.2)",
        ))

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        name="Mean EV",
        hovertemplate="episode=%{x:.0f}<br>EV=%{y:.3f}%<extra></extra>",
    ))

    target = -0.43286  # in percent units (since y is already %)
    target_label = f"Referenz ({target:.3f}%)"

    fig.add_trace(go.Scatter(
        x=[x.min(), x.max()],
        y=[target, target],
        mode="lines",
        name=target_label,
        line=dict(dash="dash", width=1, color="blue"),
        hovertemplate=f"Target: {target:.3f}%<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Episode (logarithmisch)",
        yaxis_title="Mean Return (%)",
        margin=dict(l=60, r=30, t=60, b=55),
        legend=dict(orientation="v", yanchor="bottom", y=0.1, xanchor="right", x=0.99, borderwidth=1),
    )

    fig.update_xaxes(
        type="log",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
    )

    fig.update_yaxes(
        ticksuffix="%",
        zeroline=True,
        zerolinewidth=1,
    )

    # Optional zero line (break-even)
    fig.show()
    return fig

def _bar_widths_for_log_x(x: np.ndarray, frac_of_median_spacing: float = 0.70) -> list[float]:
    """
    Plotly bar width is specified in x-data units (linear), which shrinks visually on a log x-axis.
    This computes widths so that bars have ~constant width in log-space.

    Width is chosen as a fraction of the median log10 spacing between consecutive x-values.
    """
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    if x.size < 2:
        return [0.0] * int(x.size)

    logx = np.log10(np.sort(x))
    d = np.diff(logx)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        # fallback: 5% of x
        return (0.05 * x).tolist()

    delta = float(np.median(d) * frac_of_median_spacing)  # desired full bar width in log10-units
    r = 10 ** delta

    # symmetric bar around x: log10(x+w/2) - log10(x-w/2) = delta  => w = 2x*(r-1)/(r+1)
    widths = 2.0 * x * (r - 1.0) / (r + 1.0)
    return widths.tolist()

def plot_eval_flip_rate_with_abs_flips(
    eval_history_xlsx_path: str,
    *,
    title: str = "Policy Fliprate & Anzahl Flips vs Episode",
    n_states: int = 330,
    bar_width_frac: float = 0.70,
) -> go.Figure:
    """
    Line: flip_rate (%) over train_episode (log-x).
    Bars (y2): absolute flips. Uses df['flips'] if present, else derives from flip_rate * denom/state-count.

    Bars are given widths computed to be ~constant in log-x space so they don't vanish at large episodes.
    """
    df = pd.read_excel(eval_history_xlsx_path).copy()
    if "train_episode" not in df or "flip_rate" not in df:
        raise ValueError("Expected columns: 'train_episode' and 'flip_rate'.")

    df = df.sort_values("train_episode")
    df = df[df["flip_rate"].notna()].copy()
    if df.empty:
        raise ValueError("No non-null flip_rate values found.")

    x = df["train_episode"].astype(float).to_numpy()
    flip_rate = df["flip_rate"].astype(float).to_numpy()
    flip_pct = flip_rate * 100.0

    # Prefer exact flips if you exported it; else derive
    if "flips" in df.columns and df["flips"].notna().any():
        abs_flips = df["flips"].fillna(0).astype(int).to_list()
        denom_for_title = int(df["flip_denom"].dropna().iloc[-1]) if "flip_denom" in df.columns and df["flip_denom"].notna().any() else None
    elif "flip_denom" in df.columns and df["flip_denom"].notna().any():
        denom = df["flip_denom"].astype(float).to_numpy()
        abs_flips = [int(round(fr * d)) for fr, d in zip(flip_rate, denom)]
        denom_for_title = int(np.nanmax(denom))
    else:
        abs_flips = [int(round(fr * n_states)) for fr in flip_rate]
        denom_for_title = n_states

    widths = _bar_widths_for_log_x(x, frac_of_median_spacing=bar_width_frac)

    tickvals = _nice_log_ticks(float(x.min()), float(x.max()))
    ticktext = [_fmt_si(v) for v in tickvals]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=x,
        y=abs_flips,
        width=widths,          # key change
        name="Anzahl Flips",
        opacity=0.55,
        yaxis="y2",
        hovertemplate="episode=%{x:.0f}<br>abs_flips=%{y}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=flip_pct,
        mode="lines+markers",
        name="Fliprate",
        hovertemplate="episode=%{x:.0f}<br>flip_rate=%{y:.3f}%<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Episode (logarithmisch)",
        yaxis=dict(
            title="Fliprate (%)",
            ticksuffix="%",
            rangemode="tozero",
            zeroline=True,
            zerolinewidth=1,
        ),
        yaxis2=dict(
            title=f"Anzahl Flips (von {denom_for_title})" if denom_for_title else "Absolute flips",
            overlaying="y",
            side="right",
            rangemode="tozero",
            showgrid=False,
        ),
        barmode="overlay",
        margin=dict(l=70, r=80, t=60, b=55),
        legend=dict(orientation="v", yanchor="bottom", y=0.9, xanchor="right", x=0.99, borderwidth=1),
    )

    fig.update_xaxes(
        type="log",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
    )
    fig.show()
    return fig

def plot_state_values(
    xlsx_path: str,
    *,
    sheet_name: str = "state_best",
    show: bool = True,
) -> Dict[str, go.Figure]:
    """
    Reads the `state_best` worksheet and creates 3 separate 3D Plotly figures
    (hard / soft / pair) with:
      X = dealer upcard
      Y = player hand (category-specific labels)
      Z = best_ev

    Returns a dict: {"hard": fig_hard, "soft": fig_soft, "pair": fig_pair}
    """

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    required = {"category", "label", "dealer_up", "best_ev"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in '{sheet_name}': {sorted(missing)}")

    def card_to_num(v) -> int:
        # supports 2..10 and 'A' (Ace -> 11)
        if pd.isna(v):
            raise ValueError("Found NaN in dealer_up/label.")
        s = str(v).strip().upper()
        if s == "A":
            return 11
        return int(s)

    def num_to_card(n: int) -> str:
        return "A" if n == 11 else str(int(n))

    dealer_order = list(range(2, 11)) + [11]
    dealer_tickvals = dealer_order
    dealer_ticktext = [str(x) for x in range(2, 11)] + ["A"]

    def make_surface_for_category(category: str) -> go.Figure:
        d = df[df["category"].astype(str).str.lower() == category].copy()
        if d.empty:
            raise ValueError(f"No rows found for category='{category}' in '{sheet_name}'.")

        d["dealer_up_num"] = d["dealer_up"].map(card_to_num)

        if category == "hard":
            # hard totals 5..20
            y_order = list(range(5, 21))
            d["label_num"] = d["label"].map(lambda x: int(str(x).strip()))
            y_tickvals = y_order
            y_ticktext = [str(y) for y in y_order]
            y_title = "Player hard total"
        elif category == "soft":
            # soft totals 13..20 (A2..A9)
            y_order = list(range(13, 21))
            d["label_num"] = d["label"].map(lambda x: int(str(x).strip()))
            y_tickvals = y_order
            y_ticktext = [f"A{t-11}" for t in y_order]  # 13->A2 ... 20->A9
            y_title = "Player soft hand"
        elif category == "pair":
            # pairs 2..10 and A (mapped to 11)
            y_order = list(range(2, 11)) + [11]
            d["label_num"] = d["label"].map(card_to_num)
            y_tickvals = y_order
            y_ticktext = [f"{num_to_card(v)},{num_to_card(v)}" for v in y_order]
            y_title = "Player pair"
        else:
            raise ValueError(f"Unknown category '{category}'.")

        # Pivot to a full grid (rows=y, cols=x)
        z_df = (
            d.pivot(index="label_num", columns="dealer_up_num", values="best_ev")
            .reindex(index=y_order, columns=dealer_order)
        )

        if z_df.isna().any().any():
            # If your table ever becomes sparse, you can switch to Scatter3d instead.
            raise ValueError(
                f"Category '{category}' has missing (label, dealer_up) combinations; cannot build a full surface."
            )

        x = np.array(dealer_order)
        y = np.array(y_order)
        z = z_df.to_numpy()

        # Build hover labels (show A instead of 11; show pair/soft labels nicely)
        X, Y = np.meshgrid(x, y)
        dealer_lbl = np.vectorize(num_to_card)(X)

        if category == "soft":
            player_lbl = np.vectorize(lambda t: f"A{int(t)-11}")(Y)
        elif category == "pair":
            player_lbl = np.vectorize(lambda r: f"{num_to_card(int(r))},{num_to_card(int(r))}")(Y)
        else:  # hard
            player_lbl = np.vectorize(lambda t: str(int(t)))(Y)

        customdata = np.dstack([dealer_lbl, player_lbl])

        zmin = float(np.nanmin(z))
        zmax = float(np.nanmax(z))
        absmax = max(abs(zmin), abs(zmax))

        fig = go.Figure(
            data=go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale="delta",
                cmin=-absmax + 0.1,
                cmax=absmax ,
                cmid=0.0,
                customdata=customdata,
                hovertemplate=(
                    "Dealer: %{customdata[1]}<br>"
                    "Player: %{customdata[0]}<br>"
                    "best_ev: %{z:.6f}"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=f"EV Landschaft ({category})",
            scene=dict(
                xaxis=dict(
                    title="Dealer upcard",
                    tickmode="array",
                    tickvals=dealer_tickvals,
                    ticktext=dealer_ticktext,
                ),
                yaxis=dict(
                    title=y_title,
                    tickmode="array",
                    tickvals=y_tickvals,
                    ticktext=y_ticktext,
                ),
                zaxis=dict(title="best_ev"),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        return fig

    figs = {cat: make_surface_for_category(cat) for cat in ("hard", "soft", "pair")}

    if show:
        for fig in figs.values():
            fig.show()

    return figs

def plot_epsilon(N0, eps_max, eps_min):
    Ns = np.arange(0, 22001)  # N_s = 0..12000
    e = N0 / (N0 + Ns)
    eps = np.maximum(eps_min, np.minimum(eps_max, e))

    # Breakpoints (integers, because N_s counts visits)
    Ns_break1 = int(np.floor(N0 * (1/eps_max - 1)))  # 2333  -> constant eps_max up to here
    Ns_break2 = int(N0 * (1/eps_min - 1))            # 9000  -> constant eps_min from here on

    seg1 = Ns <= Ns_break1
    seg2 = (Ns >= Ns_break1 + 1) & (Ns <= Ns_break2 - 1)  # 2334..8999
    seg3 = Ns >= Ns_break2

    fig, ax = plt.subplots(figsize=(9, 4.8))

    ax.plot(Ns[seg1], eps[seg1], linewidth=2.5, color="tab:blue",   label=f"N_s ≤ {Ns_break1} (eps_max)")
    ax.plot(Ns[seg2], eps[seg2], linewidth=2.5, color="tab:orange", label=f"{Ns_break1+1} ≤ N_s ≤ {Ns_break2-1} (decay)")
    ax.plot(Ns[seg3], eps[seg3], linewidth=2.5, color="tab:green",  label=f"N_s ≥ {Ns_break2} (eps_min)")

    # Visual markers for boundaries
    ax.axvline(Ns_break1 + 1, linestyle="--", linewidth=1.5, color="0.35")
    ax.axvline(Ns_break2,     linestyle="--", linewidth=1.5, color="0.35")

    ax.set_title(f"Epsilon(N_s) mit N0={N0}, eps_max={eps_max}, eps_min={eps_min}")
    ax.set_xlabel("N_s")
    ax.set_ylabel("Epsilon")
    ax.set_xlim(Ns[0], Ns[-1])
    ax.set_ylim(0, 0.35)
    ax.grid(True, alpha=0.5)
    ax.legend(frameon=True)

    plt.show()

    
if __name__ == "__main__":
    plot_epsilon(N0=1000, eps_max=0.3, eps_min=0.05)
    eval_history_path = r"sim_results\new_fav_0,05min_no_hard_eps\eval_history.xlsx"
    return_plot = plot_eval_return(eval_history_path)
    return_plot.write_html("EV_plot.html", include_plotlyjs="cdn", full_html=True)
    policy_flip_plot = plot_eval_flip_rate_with_abs_flips(eval_history_path)
    policy_flip_plot.write_html("Fliprate_plot.html", include_plotlyjs="cdn", full_html=True)
    state_landscape_plots = plot_state_values(r"sim_results\new_fav_0,05min_no_hard_eps\initial_decision_QN_20260118_1859.xlsx")
    for plot in state_landscape_plots.values():
        title = plot.layout.title.text
        plot.write_html(f"{title}.html", include_plotlyjs="cdn", full_html=True)
