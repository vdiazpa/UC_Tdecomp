#get_table2_results.py


import os
import glob
import pandas as pd
import numpy as np


# ----------------------------- helpers -----------------------------

def _sanitize(x):
    # matches your earlier convention: "." -> "p", "-" -> "m"
    return f"{x}".replace(".", "p").replace("-", "m")


def _find_unique_file(parent_dir, patterns):
    """Find exactly one file matching any pattern (else hard fail)."""
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(parent_dir, pat)))
    matches = sorted(set(matches))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No files matched any pattern in: {parent_dir}\nPatterns:\n  - "
            + "\n  - ".join(patterns)
        )
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple files matched (make naming/patterns more specific):\n  - "
            + "\n  - ".join(matches)
        )
    return matches[0]


def _pick_time_col(df):
    """Try common runtime column names; fail if none exist."""
    candidates = [
        "avg_time", "avg_runtime", "runtime", "rh_time",
        "time", "solve_time", "avg_solve_time"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find a runtime column. Available columns: {list(df.columns)}")


def _pick_col(df, options, label):
    """Pick a column from options (case-insensitive match); fail if none."""
    lower_map = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lower_map:
            return lower_map[opt.lower()]
    raise KeyError(f"Could not find {label} column. Tried {options}. Available: {list(df.columns)}")


def load_mono_costs_from_csv(mono_csv_path):
    """
    Reads your monolithic table CSV and builds:
      mono_costs[(SYSTEM, T)] = OFV

    Expected columns (case-insensitive):
      - System (or system)
      - T (or Horizon, horizon)
      - OFV (or ofv, obj, objective)
    """
    if not os.path.isfile(mono_csv_path):
        raise FileNotFoundError(f"Mono CSV not found: {mono_csv_path}")

    df = pd.read_csv(mono_csv_path)
    sys_col = _pick_col(df, ["System", "system"], "System")
    t_col   = _pick_col(df, ["T", "t", "Horizon", "horizon"], "T/Horizon")
    ofv_col = _pick_col(df, ["OFV", "ofv", "obj", "objective", "Objective"], "OFV/Objective")

    df[sys_col] = df[sys_col].astype(str).str.strip().str.upper()
    df[t_col]   = df[t_col].astype(int)

    mono_costs = {}
    for _, r in df.iterrows():
        key = (r[sys_col], int(r[t_col]))
        mono_costs[key] = float(r[ofv_col])

    if len(mono_costs) == 0:
        raise ValueError(f"No rows read from mono CSV: {mono_csv_path}")

    return mono_costs


# ----------------------------- main builder -----------------------------

def build_table2(
    parent_dir,
    mono_csv_path,
    systems=("RTS", "DUK"),
    Ts=(72, 168, 336),
    rh_gap=0.01,
    sweep_subdir="sweep",
    abs_cost_diff=False,   # False keeps sign (negative allowed)
    verbose=True,
):
    """
    Builds Table II-style summary from RH sweep CSVs.

    Looks for sweep CSVs under:
      parent_dir / sweep_subdir

    Expected sweep CSV columns:
      - F
      - L
      - avg_ofv
      - and a runtime column (avg_time/avg_runtime/runtime/etc.)

    Monolithic cost is read from mono_csv_path via load_mono_costs_from_csv().
    """
    parent_dir = os.path.abspath(parent_dir)
    sweep_dir = os.path.join(parent_dir, sweep_subdir)

    if not os.path.isdir(sweep_dir):
        raise NotADirectoryError(f"Missing sweep directory: {sweep_dir}")

    mono_costs = load_mono_costs_from_csv(mono_csv_path)

    rows = []
    gap_str = str(rh_gap)
    gap_san = _sanitize(rh_gap)

    for sys in systems:
        sysU = str(sys).strip().upper()
        for T in Ts:
            T = int(T)

            patterns = [
                f"rh_{sysU}_results_T{T}_gap{gap_str}*.csv",
                f"rh_{sysU}_results_T{T}_gap{gap_san}*.csv",
                f"*rh*{sysU}*T{T}*gap{gap_str}*.csv",
                f"*rh*{sysU}*T{T}*gap{gap_san}*.csv",
            ]

            fpath = _find_unique_file(sweep_dir, patterns)
            if verbose:
                print(f"[Table2] using sweep file: {fpath}")

            df = pd.read_csv(fpath)

            for c in ["F", "L", "avg_ofv"]:
                if c not in df.columns:
                    raise KeyError(f"Missing required column '{c}' in {fpath}. Columns: {list(df.columns)}")

            time_col = _pick_time_col(df)

            key = (sysU, T)
            if key not in mono_costs:
                raise KeyError(f"Mono cost missing for {key}. Check mono_csv_path and System/T values.")
            mono = float(mono_costs[key])

            sub = df[["F", "L", "avg_ofv", time_col]].copy()
            sub = sub.dropna()

            if sub.empty:
                raise ValueError(f"After dropping NaNs, no usable rows in {fpath}")

            sub["cost_diff_pct"] = 100.0 * (sub["avg_ofv"] - mono) / mono
            if abs_cost_diff:
                sub["cost_diff_pct"] = sub["cost_diff_pct"].abs()

            i_min_cd = sub["cost_diff_pct"].idxmin()
            i_max_cd = sub["cost_diff_pct"].idxmax()
            i_min_t  = sub[time_col].idxmin()
            i_max_t  = sub[time_col].idxmax()

            rows.append({
                "system": sysU,
                "T": T,

                "min_cost_diff_pct": float(sub.loc[i_min_cd, "cost_diff_pct"]),
                "min_cost_F": int(sub.loc[i_min_cd, "F"]),
                "min_cost_L": int(sub.loc[i_min_cd, "L"]),

                "max_cost_diff_pct": float(sub.loc[i_max_cd, "cost_diff_pct"]),
                "max_cost_F": int(sub.loc[i_max_cd, "F"]),
                "max_cost_L": int(sub.loc[i_max_cd, "L"]),

                "min_rh_time_s": float(sub.loc[i_min_t, time_col]),
                "min_time_F": int(sub.loc[i_min_t, "F"]),
                "min_time_L": int(sub.loc[i_min_t, "L"]),

                "max_rh_time_s": float(sub.loc[i_max_t, time_col]),
                "max_time_F": int(sub.loc[i_max_t, "F"]),
                "max_time_L": int(sub.loc[i_max_t, "L"]),

                "mono_cost": mono,
                "sweep_file": os.path.basename(fpath),
                "runtime_col": time_col,
            })

    out = pd.DataFrame(rows).sort_values(["system", "T"]).reset_index(drop=True)
    return out


# ----------------------------- example usage -----------------------------

if __name__ == "__main__":

    PARENT_DIR = r"C:\Users\vdiazpa\Documents\PCM\paper_results\results_ok_rev"
    MONO_CSV   = os.path.join(PARENT_DIR, "Mono_runtimes_MTCOunterOptGap0.01_NoChcost.csv")  # <--  mono CSV filename
    SWEEP_SUBDIR = "sweep_RH_NBFul_NoCHCost_0.01_BEnchCount"               # <--  sweep folder name
    # --------------------------

    SYSTEMS = ("RTS", "DUK")
    TS = (72, 168, 336)
    RH_GAP = 0.01

    tab2 = build_table2(
        parent_dir=PARENT_DIR,
        mono_csv_path=MONO_CSV,
        systems=SYSTEMS,
        Ts=TS,
        rh_gap=RH_GAP,
        sweep_subdir=SWEEP_SUBDIR,
        abs_cost_diff=False,   # keep negative if RH beats mono
        verbose=True,
    )

    print("\n=== TABLE 2 DATAFRAME ===")
    print(tab2.to_string(index=False))

    out_csv = os.path.join(PARENT_DIR, "Table2_summary.csv")
    tab2.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")