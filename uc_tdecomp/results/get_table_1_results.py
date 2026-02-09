#get_table_1_results.py

import os
import glob
import pandas as pd
import numpy as np


def _find_single_subdir(parent_dir, token):
    subs = []
    for d in os.listdir(parent_dir):
        p = os.path.join(parent_dir, d)
        if os.path.isdir(p) and token.lower() in d.lower():
            subs.append(p)
    if len(subs) == 0:
        raise FileNotFoundError(f"No subfolder containing '{token}' in {parent_dir}")
    if len(subs) > 1:
        raise RuntimeError(f"Multiple subfolders containing '{token}' in {parent_dir}: "
                           f"{[os.path.basename(x) for x in subs]}")
    return subs[0]


def _normalize_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return _normalize_cols(df)


def _gap_tokens(gap):
    # support common filename encodings
    s = str(gap)
    tokens = {f"gap{s}".lower()}
    if "." in s:
        tokens.add(f"gap{s.replace('.', 'p')}".lower())  # gap0p01
    return list(tokens)


def _find_single_rh_csv(sweep_dir, sys_token, T, gap):
    files = glob.glob(os.path.join(sweep_dir, "*.csv"))
    sys_l = sys_token.lower()
    t_tag = f"t{T}"
    gap_tags = _gap_tokens(gap)

    hits = []
    for p in files:
        name = os.path.basename(p).lower()
        if (sys_l in name) and (t_tag in name) and any(gt in name for gt in gap_tags):
            hits.append(p)

    if len(hits) == 0:
        raise FileNotFoundError(
            f"No RH sweep CSV matched sys={sys_token}, T={T}, gap={gap} in {sweep_dir}.\n"
            f"Available:\n  " + "\n  ".join(sorted(os.path.basename(x) for x in files))
        )
    if len(hits) > 1:
        raise RuntimeError(
            f"Multiple RH sweep CSVs matched sys={sys_token}, T={T}, gap={gap}:\n  " +
            "\n  ".join(os.path.basename(x) for x in hits)
        )
    return hits[0]


def _read_rh_cost_time_from_sweep(rh_csv_path, F, L):
    df = _normalize_cols(pd.read_csv(rh_csv_path))

    needed = {"f", "l", "avg_ofv", "avg_time"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"RH sweep file {rh_csv_path} missing columns {missing}. Has: {list(df.columns)}")

    df["f"] = pd.to_numeric(df["f"], errors="coerce")
    df["l"] = pd.to_numeric(df["l"], errors="coerce")

    rows = df[(df["f"] == F) & (df["l"] == L)]
    if len(rows) == 0:
        raise ValueError(f"No RH row with F={F}, L={L} in {rh_csv_path}")
    if len(rows) > 1:
        raise RuntimeError(f"Multiple RH rows with F={F}, L={L} in {rh_csv_path}")

    rh_cost = float(pd.to_numeric(rows["avg_ofv"], errors="coerce").iloc[0])
    rh_time = float(pd.to_numeric(rows["avg_time"], errors="coerce").iloc[0])

    if np.isnan(rh_cost) or np.isnan(rh_time):
        raise ValueError(f"RH avg_ofv/avg_time is NaN for F={F}, L={L} in {rh_csv_path}")

    return rh_cost, rh_time


def _mono_row_strict(mono_df, sys_token, T, gap=None, mt=None, rlxint=None):
    df = mono_df.copy()

    # required columns from your sheet
    for col in ["system", "t", "ofv", "runtime"]:
        if col not in df.columns:
            raise KeyError(f"Monolithic file missing required column '{col}'. Has: {list(df.columns)}")

    # filters (strict if provided)
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df[df["t"] == int(T)]

    df = df[df["system"].astype(str).str.lower() == sys_token.lower()]

    if gap is not None:
        if "gap" not in df.columns:
            raise KeyError("Monolithic file has no 'gap' column to filter on.")
        df["gap"] = pd.to_numeric(df["gap"], errors="coerce")
        df = df[df["gap"] == float(gap)]

    if mt is not None:
        if "mt" not in df.columns:
            raise KeyError("Monolithic file has no 'mt' column to filter on.")
        df = df[df["mt"].astype(str).str.lower() == str(mt).lower()]

    if rlxint is not None:
        if "rlxint" not in df.columns:
            raise KeyError("Monolithic file has no 'rlxint' column to filter on.")
        # accept TRUE/FALSE strings or booleans
        v = df["rlxint"]
        if v.dtype == bool:
            df = df[df["rlxint"] == bool(rlxint)]
        else:
            df = df[df["rlxint"].astype(str).str.lower() == str(rlxint).lower()]

    if len(df) == 0:
        raise ValueError(f"No mono row matched sys={sys_token}, T={T} with filters gap={gap}, mt={mt}, rlxint={rlxint}")
    if len(df) > 1:
        raise RuntimeError(f"Multiple mono rows matched sys={sys_token}, T={T} with filters gap={gap}, mt={mt}, rlxint={rlxint}:\n{df}")

    return df.iloc[0]


def _system_label(sys_token):
    s = sys_token.upper()
    if s in ["DUK", "DUKE", "REGIONAL"]:
        return "Regional"
    if s in ["RTS", "RTS-GMLC", "RTS_GMLC", "RTSGMLC"]:
        return "RTS-GMLC"
    return sys_token


def build_table1_strict(parent_dir,
                        mono_path,
                        Ts=(72, 168, 336),
                        systems=("RTS", "DUK"),
                        F=24,
                        L=12,
                        gap=0.01,
                        mono_gap=0.01,
                        mono_mt="Counter",
                        mono_rlxint=True,
                        cost_scale=1e6,          # divide dollars -> M$
                        print_sources=True,
                        out_csv=None):
    """
    Table 1: RH performance vs monolithic

    Output columns:
      System, Horizon (h), RH Cost (M$), Mono Cost (M$), Cost diff (%), RH (s), Mono (s)

    Strict: no fallback; ambiguity/missing -> error; prints filenames used.
    """
    sweep_root = _find_single_subdir(parent_dir, "sweep")
    mono_df = _read_table(mono_path)

    rows = []
    for sys_token in systems:
        for T in Ts:
            # --- Monolithic ---
            mr = _mono_row_strict(mono_df, sys_token=sys_token, T=T,
                                  gap=mono_gap, mt=mono_mt, rlxint=mono_rlxint)
            mono_cost = float(pd.to_numeric(mr["ofv"], errors="coerce"))
            mono_time = float(pd.to_numeric(mr["runtime"], errors="coerce"))

            if np.isnan(mono_cost) or np.isnan(mono_time):
                raise ValueError(f"Mono OFV/Runtime is NaN for sys={sys_token}, T={T} in {mono_path}")

            # --- RH (from sweep file at F,L) ---
            rh_csv = _find_single_rh_csv(sweep_root, sys_token=sys_token, T=T, gap=gap)
            rh_cost, rh_time = _read_rh_cost_time_from_sweep(rh_csv, F=F, L=L)

            # scale to M$ (matches your paper table style)
            rh_cost_M = rh_cost / cost_scale
            mono_cost_M = mono_cost / cost_scale

            cost_diff_pct = 100.0 * (rh_cost_M - mono_cost_M) / (abs(mono_cost_M) if abs(mono_cost_M) > 1e-12 else 1.0)

            if print_sources:
                print("\n--- Table 1 sources ---")
                print(f"sys={sys_token}, T={T}")
                print(f"  MONO file: {mono_path}")
                print(f"    -> OFV={mono_cost}  Runtime={mono_time}")
                print(f"  RH file:   {rh_csv}")
                print(f"    -> avg_ofv={rh_cost}  avg_time={rh_time} at F={F}, L={L}")
                print(f"  (scaled) RH={rh_cost_M:.4f} M$, MONO={mono_cost_M:.4f} M$, diff={cost_diff_pct:.4f}%")

            rows.append({
                "System": _system_label(sys_token),
                "Horizon (h)": int(T),
                "RH Cost (M$)": rh_cost_M,
                "Mono Cost (M$)": mono_cost_M,
                "Cost diff (%)": cost_diff_pct,
                "RH (s)": float(rh_time),
                "Mono (s)": float(mono_time),
            })

    table = pd.DataFrame(rows).sort_values(["System", "Horizon (h)"]).reset_index(drop=True)

    if out_csv is not None:
        table.to_csv(out_csv, index=False)

    return table


# Example:
parent = r"C:\Users\vdiazpa\Documents\PCM\paper_results\final_ok_rev"
mono   = os.path.join(parent, "Mono_runtimes_MTCOunterOptGap0.01_NoChcost.xlsx")  # or .csv
t1 = build_table1_strict(parent, mono, systems=("DUK","RTS"), Ts=(72,168,336),
                         F=24, L=12, gap=0.01, mono_gap=0.01, mono_mt="Counter", mono_rlxint=True,
                         out_csv=os.path.join(parent, "Table1.csv"))
print(t1)