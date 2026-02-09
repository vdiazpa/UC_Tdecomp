#get_table3_results.py

import os
import glob
import re
import pandas as pd
import numpy as np


def _find_single_subdir(parent_dir, token):
    """Return exactly one subdirectory whose name contains token (case-insensitive)."""
    subs = []
    for d in os.listdir(parent_dir):
        p = os.path.join(parent_dir, d)
        if os.path.isdir(p) and token.lower() in d.lower():
            subs.append(p)
    if len(subs) == 0:
        raise FileNotFoundError("No subfolder containing '{}' in {}".format(token, parent_dir))
    if len(subs) > 1:
        raise RuntimeError("Multiple subfolders containing '{}' in {}: {}".format(
            token, parent_dir, [os.path.basename(x) for x in subs]
        ))
    return subs[0]

def _find_single_lr_run_folder(lr_root, sys_token, T):
    """Find exactly one LR run folder under lr_root that matches sys_token and T."""
    hits = []
    for d in os.listdir(lr_root):
        p = os.path.join(lr_root, d)
        if not os.path.isdir(p):
            continue
        name = d.lower()
        if sys_token.lower() in name and str(T) in name:
            hits.append(p)
    if len(hits) == 0:
        raise FileNotFoundError("No LR run folder found for sys={}, T={} under {}".format(sys_token, T, lr_root))
    if len(hits) > 1:
        raise RuntimeError("Multiple LR run folders found for sys={}, T={}: {}".format(
            sys_token, T, [os.path.basename(x) for x in hits]
        ))
    return hits[0]

def _parse_lr_window_from_filename(fname):
    """Extract LR window size from '_W##' in the LR filename."""
    m = re.search(r"_W(\d+)", fname)
    if not m:
        raise ValueError("Could not parse LR window size from filename (missing _W##): {}".format(fname))
    return int(m.group(1))

def _find_lr_csv_optional(lr_run_dir, T, W, gamma_int, ghat_int=1):
    """
    Find LR csv for a specific window W. Return path if found, else None.
    Looks for filename with: _T{T}_W{W}_g0p{gamma_int}_ghat{ghat_int}.csv (order flexible).
    """
    gtag = f"g0p{gamma_int}"
    if ghat_int is None:
        pat = os.path.join(lr_run_dir, f"lr_*T{T}*_W{W}_*{gtag}*.csv")
    else:
        pat = os.path.join(lr_run_dir, f"lr_*T{T}*_W{W}_*{gtag}*ghat{ghat_int}*.csv")

    hits = sorted(glob.glob(pat))
    if len(hits) == 0:
        return None
    if len(hits) > 1:
        raise RuntimeError(f"Multiple LR CSVs matched for W={W}:\n  " +
                           "\n  ".join(os.path.basename(x) for x in hits))

    cols = pd.read_csv(hits[0], nrows=1).columns
    if "best_dual" not in cols:
        raise KeyError(f"'best_dual' not found in {hits[0]}. Columns={list(cols)}")

    return hits[0]


def _read_lr_lb_last_best_dual(lr_csv_path):
    """LB = last non-NaN value in best_dual."""
    df = pd.read_csv(lr_csv_path)
    s = pd.to_numeric(df["best_dual"], errors="coerce").dropna()
    if s.empty:
        raise ValueError("No numeric values in best_dual for {}".format(lr_csv_path))
    return float(s.iloc[-1])

def _find_single_rh_csv(sweep_dir, sys_token, T, gap):
    """
    Strictly find exactly one RH sweep CSV.
    Looks for sys token + T + gap substring in filename.
    """
    files = glob.glob(os.path.join(sweep_dir, "*.csv"))
    sys_l = sys_token.lower()
    t_tag = "t{}".format(T)
    gap_tag = "gap{}".format(gap)  # expects gap0.01 style

    hits = []
    for p in files:
        name = os.path.basename(p).lower()
        if (sys_l in name) and (t_tag in name) and (gap_tag in name):
            hits.append(p)

    if len(hits) == 0:
        existing = sorted(os.path.basename(x) for x in files)
        raise FileNotFoundError(
            "No RH CSV matched sys={}, T={}, gap={} in {}\nExisting:\n  {}".format(
                sys_token, T, gap, sweep_dir, "\n  ".join(existing)
            )
        )
    if len(hits) > 1:
        raise RuntimeError("Multiple RH CSVs matched:\n  {}".format("\n  ".join(os.path.basename(x) for x in hits)))

    cols = pd.read_csv(hits[0], nrows=1).columns
    need = set(["F", "L", "avg_ofv"])
    if not need.issubset(set(cols)):
        raise KeyError("RH CSV {} missing columns {}. Columns={}".format(
            hits[0], list(need - set(cols)), list(cols)
        ))

    return hits[0]

def _read_rh_ub_avg_ofv(rh_csv_path, F, L):
    """UB = avg_ofv in row where (F,L) matches exactly one row."""
    df = pd.read_csv(rh_csv_path)
    df["F"] = pd.to_numeric(df["F"], errors="coerce")
    df["L"] = pd.to_numeric(df["L"], errors="coerce")

    rows = df[(df["F"] == F) & (df["L"] == L)]
    if len(rows) == 0:
        raise ValueError("No row with F={}, L={} in {}".format(F, L, rh_csv_path))
    if len(rows) > 1:
        raise RuntimeError("Multiple rows with F={}, L={} in {}".format(F, L, rh_csv_path))

    ub = pd.to_numeric(rows["avg_ofv"], errors="coerce").iloc[0]
    if pd.isna(ub):
        raise ValueError("avg_ofv is NaN for F={}, L={} in {}".format(F, L, rh_csv_path))
    return float(ub)

def _system_label(sys_token):
    s = sys_token.upper()
    if s in ["DUK", "DUKE", "REGIONAL"]:
        return "Regional"
    if s in ["RTS", "RTS-GMLC", "RTS_GMLC", "RTSGMLC"]:
        return "RTS-GMLC"
    return sys_token


def build_table3_windows(parent_dir,
                         Ts=(72, 168, 336),
                         systems=("DUK", "RTS"),
                         lr_windows=(24, 48),
                         gamma_int=4,
                         ghat_int=1,
                         F=24,
                         L=12,
                         gap=0.01,
                         print_sources=True,
                         out_csv=None,
                         na_token="--"):
    """
    Wide Table III:
      System, Horizon, RH UB,
      LR LB (W=24), Gap% (W=24), LR LB (W=48), Gap% (W=48), ...

    Missing LR files => NaN internally; you can print/save with na_token.
    """
    lr_root = _find_single_subdir(parent_dir, "lr")
    sweep_root = _find_single_subdir(parent_dir, "sweep")

    rows = []
    for sys_token in systems:
        for T in Ts:
            # RH UB (same regardless of LR window)
            rh_csv = _find_single_rh_csv(sweep_root, sys_token, T, gap)
            ub = _read_rh_ub_avg_ofv(rh_csv, F, L)

            lr_run_dir = _find_single_lr_run_folder(lr_root, sys_token, T)

            row = {
                "System": _system_label(sys_token),
                "Horizon": int(T),
                "RH UB": ub,
            }

            if print_sources:
                print("\n--- Table 3 sources ---")
                print(f"sys={sys_token}, T={T}")
                print("  RH:", rh_csv)
                print(f"    -> UB(avg_ofv @ F={F}, L={L})={ub}")

            # For each LR window, try to locate its csv
            for W in lr_windows:
                lr_csv = _find_lr_csv_optional(lr_run_dir, T, W, gamma_int, ghat_int=ghat_int)
                col_lb = f"LR LB (W={W})"
                col_gp = f"Gap (%) (W={W})"

                if lr_csv is None:
                    row[col_lb] = np.nan
                    row[col_gp] = np.nan
                    if print_sources:
                        print(f"  LR: [MISSING] W={W} (expected _W{W}_ in filename)")
                else:
                    lb = _read_lr_lb_last_best_dual(lr_csv)
                    gap_pct = 100.0 * (ub - lb) / (abs(ub) if abs(ub) > 1e-12 else 1.0)
                    row[col_lb] = lb
                    row[col_gp] = gap_pct

                    if print_sources:
                        print("  LR:", lr_csv)
                        print(f"    -> LB(last best_dual)={lb}")
                        print(f"    -> Gap%={gap_pct:.4f}")

            rows.append(row)

    table = pd.DataFrame(rows).sort_values(["System", "Horizon"]).reset_index(drop=True)

    # Optional: write CSV with placeholders for missing
    if out_csv is not None:
        table_to_save = table.copy()
        table_to_save = table_to_save.replace({np.nan: na_token})
        table_to_save.to_csv(out_csv, index=False)

    return table



parent = "/Users/veronica126/Documents/final_ok_rev"
tbl = build_table3_windows(
    parent,
    Ts=(72,168,336),
    systems=("DUK","RTS"),
    lr_windows=(24,48),
    gamma_int=4, ghat_int=1,
    F=24, L=12, gap=0.01,
    out_csv=os.path.join(parent, "Table3_W24_W48.csv"),
    na_token="--"
)
print(tbl)
