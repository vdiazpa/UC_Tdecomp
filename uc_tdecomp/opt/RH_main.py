
from .RH_subp_build import build_RH_subprobs
from ..data.data_extract import load_uc_data, load_csv_data
from .bench_UC import benchmark_UC_build
from time import perf_counter
import pandas as pd
import numpy as np
import csv


def RH_windows_fixes(T, F, L):
    #Gets window lengths and fixed time periods given L, F, and T
    W = F + L
    t = 1
    windows = []
    fixes = []

    t_star = T - W + 1
    while t <= T:
        r = T - t + 1
        H = min(W, r)
        s_e = list(range(t, t + H))
        windows.append(s_e)

        # alignment guard: roll less so NEXT start hits s_star
        w_o = (t + W - 1 > T)
        w_s = (t < t_star) and (t + min(F, H) > t_star)

        if (t < t_star) and (w_o or w_s):
            F_k = min(F, t_star - t, H)
        else:
            F_k = min(F, H)

        #final aligned window: if next start is >= t_star, freeze whole H
        if t >= t_star:
            fixes.append((t, t + H - 1))
            t += H
        else:
            fixes.append((t, t + F_k - 1))
            t += F_k

    return windows, fixes


def run_RH(data, F, L, T, write_csv, opt_gap, verbose, benchmark=False, seed=None, RH_opt_gap= 0.01):
    # Will run the rolling horizon algorithm given T, L, and F. 
    if T is None: 
        T = max(data["periods"])

    windows, fixes = RH_windows_fixes(T, F, L)
    fixed_sol   = {"UnitOn":{}, "UnitStart":{}, "UnitStop":{}, 'IsCharging':{}, 'IsDischarging':{}, 'SoC':{}, 'ChargePower':{}, 'DischargePower':{}}
    warm_start  = None
    init_states = {}

    if verbose:
        print(f"Running RH with F = {F}, L = {L}, T = {T}")

    t0 = perf_counter()

    for i, (window, fix_periods) in enumerate(zip(windows, fixes)):
        t_fix0, t_fix1 = fix_periods
        
        if verbose:
            print(f"Window {i+1}/{len(windows)}: {window} | fix {fix_periods}")

        result = build_RH_subprobs(data, window, init_states if i>0 else {}, fix_periods,  warm_start = warm_start, RH_opt_gap=RH_opt_gap)

        warm_start  = result["warm_start"]
        init_states = result["InitialState"]

        for k in fixed_sol.keys():
            for (g,t), v in result["vars"][k].items():
                if t_fix0 <= t <= t_fix1:
                    fixed_sol[k][(g,t)] = float(v) if k in ["PowerGenerated", "SoC", "ChargePower", "DischargePower"] else int(round(v))

    rh_time = perf_counter() - t0
    
    if verbose:
        print(f"\nTotal RH time: {rh_time:.3f} secs")

    eval_res = benchmark_UC_build(data, opt_gap, fixed_commitment = fixed_sol, F=F, L=L)
    ofv      = eval_res.get("ofv", None)

    if write_csv:           # Collect all time periods and generators
        all_t = sorted({t for (g, t) in fixed_sol["UnitOn"].keys()})
        all_g = sorted({g for (g, t) in fixed_sol["UnitOn"].keys()})
        all_b = sorted({b for (b, t) in fixed_sol['IsCharging'].keys()})

        with open(f"RHcommits_F{F}_L{L}_T{T}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Variable", "Generator"] + all_t)
            for g in all_g:
                row = ["UnitOn", g]
                for t in all_t:
                    row.append(fixed_sol["UnitOn"].get((g, t), ""))
                writer.writerow(row)
            for g in all_g:
                row = ["UnitStart", g]
                for t in all_t:
                    row.append(fixed_sol["UnitStart"].get((g, t), ""))
                writer.writerow(row)
            for g in all_g:
                row = ["UnitStop", g]
                for t in all_t:
                    row.append(fixed_sol["UnitStop"].get((g, t), ""))
                writer.writerow(row)
            for b in all_b:
                row = ["IsCharging", b]
                for t in all_t:
                    row.append(fixed_sol['IsCharging'].get((b, t), ""))
                writer.writerow(row)
            for b in all_b:
                row = ["IsDischarging", b]
                for t in all_t:
                    row.append(fixed_sol['IsDischarging'].get((b, t), ""))
                writer.writerow(row)
            for b in all_b:
                row = ["ChargePower", b]
                for t in all_t:
                    row.append(fixed_sol['ChargePower'].get((b, t), ""))
                writer.writerow(row)
            for b in all_b:
                row = ["DischargePower", b]
                for t in all_t:
                    row.append(fixed_sol['DischargePower'].get((b, t), ""))
                writer.writerow(row)
            for b in all_b:
                row = ["SoC", b]
                for t in all_t:
                    row.append(fixed_sol['SoC'].get((b, t), ""))
                writer.writerow(row)
                
    if benchmark:
        benchmark_UC_build(data, opt_gap)

    return rh_time, ofv, fixed_sol


def sweep_RH(data, T =4, F_vals = [12,24], L_vals = [8,12], seeds=(41, 86, 55), opt_gap = 0.01, only_valid = False, csv_path = f"rh_duke_results_EXP_4HR_sto.csv", verbose = False):

    records = []
    for F in F_vals:
        for L in L_vals:
            if only_valid and (F + L > T):
                continue

            times, ofvs = [], []
            for s in seeds:
                try:
                    rh_time, ofv, _ = run_RH(
                        data, F, L, T, opt_gap=opt_gap, write_csv=False, verbose=verbose and (s == seeds[0]))
                    times.append(rh_time)
                    ofvs.append(ofv)
                except Exception as e:
                    # keep going; record failure for this seed
                    if verbose:
                        print(f"  (F={F}, L={L}, seed={s}) failed: {e}")
                    times.append(np.nan)
                    ofvs.append(np.nan)

            # summarize
            rec = dict(
                F=F, L=L,
                avg_time=np.nanmean(times),
                std_time=np.nanstd(times),
                min_time=np.nanmin(times),
                max_time=np.nanmax(times),
                avg_ofv=np.nanmean(ofvs),
                std_ofv=np.nanstd(ofvs),
                seed1_time=times[0] if len(times) > 0 else np.nan,
                seed2_time=times[1] if len(times) > 1 else np.nan,
                seed3_time=times[2] if len(times) > 2 else np.nan,
                seed1_ofv=ofvs[0] if len(ofvs) > 0 else np.nan,
                seed2_ofv=ofvs[1] if len(ofvs) > 1 else np.nan,
                seed3_ofv=ofvs[2] if len(ofvs) > 2 else np.nan,
)
            records.append(rec)

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["F","L"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path}")
    return df

