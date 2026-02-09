#lr_plots.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _fmt_tag(x):
    if isinstance(x, str):
        return x
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    s = f"{x:g}"
    return s.replace(".", "p").replace("-", "m")

def plot_lr_gap_and_dual(base_dir, system, T, W, gammas, gamma_hat=1,
                         out_dir="Plots",
                         figsize=(3.5, 2.6),
                         save_svg=True,
                         dpi_png=600,
                         y_gap_max=None):
    """
    Two-panel LR figure:
      (top) gap% vs iteration  [thin=per-iter, thick=running best]
      (bot) dual LB vs iteration [thin=per-iter, thick=running best]

    Expects files:
      lr_{system}_T{T}_W{W}_g{gTag}_ghat{ghatTag}.csv

    Uses:
      UB = df['level'].iloc[0]
      lb_inst = df['dual_lb']
      lb_best = running max(lb_inst)
      gap_inst = 100*(UB - lb_inst)/UB
      gap_best = 100*(UB - lb_best)/UB
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "Times New Roman"})

    ghat_tag = _fmt_tag(gamma_hat)

    # Read all runs first (so we can pick a nice dual scaling)
    runs = []
    for g in gammas:
        g_tag = _fmt_tag(g)
        fname = f"lr_{system}_T{T}_W{W}_g{g_tag}_ghat{ghat_tag}.csv"
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)
        for col in ["iteration", "dual_lb", "level"]:
            if col not in df.columns:
                raise ValueError(f"{fname} missing column '{col}'. Has: {df.columns.tolist()}")

        it = df["iteration"].to_numpy()
        UB = float(df["level"].iloc[0])
        lb_inst = df["dual_lb"].to_numpy(dtype=float)
        lb_best = np.maximum.accumulate(lb_inst)

        gap_inst = 100.0 * (UB - lb_inst) / UB
        gap_best = 100.0 * (UB - lb_best) / UB

        runs.append(dict(g=g, it=it, UB=UB,
                         lb_inst=lb_inst, lb_best=lb_best,
                         gap_inst=gap_inst, gap_best=gap_best))

    # Dual scaling so the bottom axis is readable (billions -> nicer ticks)
    # If UB is big, scale to 1e9; otherwise no scaling.
    UB0 = abs(runs[0]["UB"])
    dual_scale = 1e9 if UB0 >= 1e8 else 1.0
    dual_label = "Dual LB (×10⁹)" if dual_scale == 1e9 else "Dual LB"

    # Figure with two stacked panels
    fig, (ax_gap, ax_dual) = plt.subplots(2, 1, sharex=True, figsize=figsize)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.14, hspace=0.35)

    # Plot each gamma in a consistent color (matplotlib cycle)
    for r in runs:
        g = r["g"]
        it = r["it"]

        # TOP: gap%
        ax_gap.plot(it, r["gap_inst"], alpha=0.18, linewidth=0.8, label="_nolegend_")
        ax_gap.plot(it, r["gap_best"], linewidth=1.4, label=f"γ={g}")

        # BOTTOM: dual LB
        ax_dual.plot(it, r["lb_inst"]/dual_scale, alpha=0.18, linewidth=0.8, label="_nolegend_")
        ax_dual.plot(it, r["lb_best"]/dual_scale, linewidth=1.4, label=f"γ={g}")

    # Cosmetics: axes labels/ticks
    ax_gap.set_ylabel("Gap (%)", fontsize=8)
    ax_dual.set_ylabel(dual_label, fontsize=8)
    ax_dual.set_xlabel("Iteration", fontsize=8)

    for ax in (ax_gap, ax_dual):
        ax.tick_params(axis="both", which="major", labelsize=7)

    # Optional fixed y-lims for gap (helps comparisons across systems)
    if y_gap_max is not None:
        ax_gap.set_ylim(0, y_gap_max)

    # Legend: only gamma (thick lines), put it in the top panel
    ax_gap.legend(ncol=1, fontsize=6, loc="upper right",
                  frameon=True, fancybox=True, framealpha=0.7,
                  borderpad=0.25, handlelength=1.2, handletextpad=0.3,
                  labelspacing=0.2, borderaxespad=0.3)

    # Tiny style note (so reviewers know thin vs thick)
    ax_gap.text(0.02, 0.02, "thin: per-iteration   thick: running best",
                transform=ax_gap.transAxes, fontsize=7, va="bottom")

    # Save
    base = f"lr_gap_dual_{system}_T{T}_W{W}_ghat{ghat_tag}"
    out_pdf = os.path.join(out_dir, base + ".pdf")
    fig.savefig(out_pdf, bbox_inches="tight")  # vector (best)

    if save_svg:
        out_svg = os.path.join(out_dir, base + ".svg")
        fig.savefig(out_svg, bbox_inches="tight")

    out_png = os.path.join(out_dir, base + ".png")
    fig.savefig(out_png, dpi=dpi_png, bbox_inches="tight")

    plt.close(fig)
    print("Saved:", out_pdf)
    if save_svg:
        print("Saved:", out_svg)
    print("Saved:", out_png)

    return out_pdf

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _fmt_tag(x):
    if isinstance(x, str):
        return x
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    s = f"{x:g}"
    return s.replace(".", "p").replace("-", "m")
# lr_plots.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _fmt_tag(x):
    """0.2 -> '0p2', 1 -> '1' (matches your filenames)."""
    if isinstance(x, str):
        return x
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    s = f"{x:g}"
    return s.replace(".", "p").replace("-", "m")


def plot_lr_gap_only(
    base_dir,
    system,
    T,
    W,
    gammas,
    gamma_hat=1,
    out_dir="Plots",
    figsize=(3.5, 1.55),   # IEEE 2-col friendly
    save_svg=True,
    dpi_png=600,
    x_max=15,              # crop to show “action”
    y_gap_min=None,
    y_gap_max=None,
    show_inst=True,        # faint oscillations (optional)
    legend_loc="upper right",
    palette=("navy", "teal", "gold"),  # <- requested colors
):
    """
    Single-panel LR figure:
      Gap(%) vs Iteration for multiple gammas.
      Thick = running-best gap (monotone).
      Optional thin = per-iteration gap (faint oscillations).

    Expects CSVs named:
      lr_{system}_T{T}_W{W}_g{gTag}_ghat{ghatTag}.csv

    Requires columns: iteration, dual_lb, level
      UB = level at iteration 0 (df['level'].iloc[0])
      LB_k = dual_lb at iter k; best LB = running max
      gap_k = 100*(UB - LB_k)/UB
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "Times New Roman"})

    # --- fixed palette (navy / teal / gold)
    name_to_hex = {
        "navy": "#0B1F3B",
        "teal": "#0F766E",
        "gold": "#D4A017",
    }
    colors = [name_to_hex.get(c.lower(), c) for c in palette]

    ghat_tag = _fmt_tag(gamma_hat)

    fig, ax = plt.subplots(figsize=figsize)

    for i, g in enumerate(gammas):
        c = colors[i % len(colors)]
        g_tag = _fmt_tag(g)
        fname = f"lr_{system}_T{T}_W{W}_g{g_tag}_ghat{ghat_tag}.csv"

        # strict path (preferred)
        path = os.path.join(base_dir, fname)

        # fallback: glob in case of tiny naming differences
        if not os.path.exists(path):
            pat = os.path.join(base_dir, f"lr_*{system}*T{T}*W{W}*g{g_tag}*ghat{ghat_tag}*.csv")
            hits = sorted(glob.glob(pat))
            if len(hits) == 0:
                raise FileNotFoundError(f"Missing file:\n  {path}\n(no match for glob)\n  {pat}")
            if len(hits) > 1:
                raise RuntimeError("Multiple files matched:\n  " + "\n  ".join(hits))
            path = hits[0]

        df = pd.read_csv(path)
        for col in ["iteration", "dual_lb", "level"]:
            if col not in df.columns:
                raise ValueError(f"{os.path.basename(path)} missing '{col}'. Has: {df.columns.tolist()}")

        it = df["iteration"].to_numpy()
        UB = float(df["level"].iloc[0])

        lb_inst = df["dual_lb"].to_numpy(dtype=float)
        lb_best = np.maximum.accumulate(lb_inst)

        gap_inst = 100.0 * (UB - lb_inst) / UB
        gap_best = 100.0 * (UB - lb_best) / UB

        if show_inst:
            ax.plot(it, gap_inst, color=c, alpha=0.18, linewidth=0.8, label="_nolegend_")
        ax.plot(it, gap_best, color=c, linewidth=1.6, label=fr"$\gamma={g}$")

    ax.set_xlabel("Iteration", fontsize=8)
    ax.set_ylabel("Gap (\%)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=7)

    # crop to show action
    if x_max is not None:
        ax.set_xlim(0, x_max)

    # y-range control (optional)
    if (y_gap_min is not None) or (y_gap_max is not None):
        lo = y_gap_min if y_gap_min is not None else ax.get_ylim()[0]
        hi = y_gap_max if y_gap_max is not None else ax.get_ylim()[1]
        ax.set_ylim(lo, hi)

    ax.legend(
        ncol=1, fontsize=6, loc=legend_loc,
        frameon=True, fancybox=True, framealpha=0.7,
        borderpad=0.25, handlelength=1.2, handletextpad=0.3,
        labelspacing=0.2, borderaxespad=0.3
    )

    plt.tight_layout()

    base = f"lr_gap_{system}_T{T}_W{W}_ghat{ghat_tag}"
    out_pdf = os.path.join(out_dir, base + ".pdf")
    fig.savefig(out_pdf, bbox_inches="tight")

    if save_svg:
        out_svg = os.path.join(out_dir, base + ".svg")
        fig.savefig(out_svg, bbox_inches="tight")

    out_png = os.path.join(out_dir, base + ".png")
    fig.savefig(out_png, dpi=dpi_png, bbox_inches="tight")

    plt.close(fig)
    print("Saved:", out_pdf)
    if save_svg:
        print("Saved:", out_svg)
    print("Saved:", out_png)

    return out_pdf


# -------- Example usage --------
if __name__ == "__main__":
    plot_lr_gap_only(
        base_dir="/Users/veronica126/Documents/final_ok_rev/LR_RUNS_ok/LR_runs_RTS_72",
        system="RTS",
        T=72,
        W=24,
        gammas=[0.2, 0.4, 0.6],
        gamma_hat=1,
        out_dir="/Users/veronica126/Documents/final_ok_rev/LR_RUNS_ok/LR_runs_RTS_72/Plots",
        x_max=15,
        show_inst=True,       # set False if you want ONLY the thick lines
        palette=("navy", "teal", "gold"),
    )



# # -------- Example usage --------
# if __name__ == "__main__":
#     plot_lr_gap_only(
#         base_dir="/Users/veronica126/Documents/final_ok_rev/LR_RUNS_ok/LR_runs_RTS_72",
#         system="RTS",
#         T=72,
#         W=24,
#         gammas=[0.2, 0.4, 0.6],
#         gamma_hat=1,
#         out_dir="/Users/veronica126/Documents/final_ok_rev/LR_RUNS_ok/LR_runs_RTS_72/Plots",
#         x_max=15,         # key improvement
#         show_inst=True,   # optional; set False if you want cleaner
#         y_gap_min=None,
#         y_gap_max=None    # or set e.g. 12 if you want fixed across plots
#     )


# plot_lr_gap_only(
#     base_dir="/Users/veronica126/Documents/final_ok_rev/LR_RUNS_ok/LR_runs_RTS_72",
#     system="RTS",
#     T=72,
#     W=24,
#     gammas=[0.2, 0.4, 0.6],
#     gamma_hat=1,
#     out_dir="Plots",
#     y_gap_max=12,
#     show_inst=False
# )


# plot_lr_gap_and_dual(
#     base_dir="/Users/veronica126/Documents/LR_RUNS_ok/LR_runs_DUK_72",
#     system="DUK",
#     T=72,
#     W=24,
#     gammas=[0.2, 0.4, 0.6],
#     gamma_hat=1,
#     out_dir="Plots",
#     y_gap_max=12  # optional; pick something that fits your gaps
# )
