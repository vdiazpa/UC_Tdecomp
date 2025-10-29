
from uc_tdecomp.RH_subp_build_t import build_RH_subprobs_t
from uc_tdecomp.data_extract import load_uc_data, load_csv_data
from uc_tdecomp.bench_UC import benchmark_UC_build
from time import perf_counter
import pandas as pd
import numpy as np
import csv

L = 4            # Lookahead
F = 8            # Roll forward period
T = 168          # length of planning horizon
prt_cry = False  # Print carryover constraints
opt_gap = 0.05   # Optimality gap for monolithic solve

################################### Load data #####################################
#file_path  = "examples/unit_commitment/RTS_GMLC_zonal_noreserves.json"
#file_path  = "examples/unit_commitment/tiny_RTS_ready.json"
#data = load_uc_data(file_path)
#data =  load_csv_data(T)
###################################################################################

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


def run_RH(data, F, L, T, write_csv, opt_gap, verbose, benchmark=False, seed=None):

    if T is None: 
        T = max(data["periods"])

    windows, fixes = RH_windows_fixes(T, F, L)
    fixed_sol   = {"UnitOn":{}, "UnitStart":{}, "UnitStop":{}}
    init_states = {}
    warm_start = None

    if verbose:
        print(f"Running RH with F = {F}, L = {L}, T = {T}")

    t0 = perf_counter()

    for i, (window, fix_periods) in enumerate(zip(windows, fixes)):
        t_fix0, t_fix1 = fix_periods
        if verbose:
            print(f"Window {i+1}/{len(windows)}: {window} | fix {fix_periods}")

        result = build_RH_subprobs_t(data, window, init_states if i>0 else {}, fix_periods, print_carryover = prt_cry, 
                                     opt_gap = opt_gap, warm_start = warm_start, solver_seed=seed)

        warm_start  = result["warm_start"]
        init_states = result["InitialState"]

        for k in fixed_sol.keys():
            for (g,t), v in result["vars"][k].items():
                if t_fix0 <= t <= t_fix1:
                    fixed_sol[k][(g,t)] = float(v) if k == "PowerGenerated" else int(round(v))

    rh_time = perf_counter() - t0
    
    if verbose:
        print(f"\nTotal RH time: {rh_time:.3f} secs")

    eval_res = benchmark_UC_build(data, opt_gap, fixed_commitment = fixed_sol)
    ofv      = eval_res.get("ofv", None)

    if write_csv:           # Collect all time periods and generators
        all_t = sorted({t for (g, t) in fixed_sol["UnitOn"].keys()})
        all_g = sorted({g for (g, t) in fixed_sol["UnitOn"].keys()})

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
                
    if benchmark:
        benchmark_UC_build(data, opt_gap)

    return rh_time, ofv, fixed_sol

# commitment, ofv, _ = run_RH(data, F = F, L = L, T = T, write_csv = True, opt_gap = opt_gap, verbose = True, benchmark=True)
# print("OFV with RH according to function is: ", ofv)


def sweep_RH(data, T =T, F_vals = [4,8,12,16,20,24], L_vals = [4,8,12,16,20,24], seeds=(41,72,36, 65, 797), opt_gap = opt_gap, only_valid = False, csv_path = "rh_duke_results_EXP1.csv", verbose = False):

    records = []
    for F in F_vals:
        for L in L_vals:
            if only_valid and (F + L > T):
                continue

            times, ofvs = [], []
            for s in seeds:
                try:
                    rh_time, ofv, _ = run_RH(
                        data, F, L, T,
                        opt_gap=opt_gap,
                        write_csv=False,
                        verbose=verbose and (s == seeds[0]),  # print once per (F,L) 
                        seed=s
                    )
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

#df = sweep_RH( data, T = T, F_vals = [4,8,12,16,20,24], L_vals = [4,8,12,16,20,24], seeds=(41,72,36, 65, 797), opt_gap = 0.05, only_valid = True, csv_path = "rh_duke_results_EXP2_168HR.csv", verbose = True)

##############Plotting code##############

# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# csv_path = r"C:\Users\vdiazpa\Documents\Egret\rh_T36_seed1.csv"

# df = pd.read_csv(csv_path)

# x   = df["F"]
# y   = df["L"]
# ofv = df["avg_ofv"]   
# runtime = df["avg_time"]
# target = ofv

# fig = plt.figure()
# ax  = fig.add_subplot(111, projection='3d')

# ax.scatter(x, y, target, c=target, cmap='viridis', s=50)
# ax.set_xlim(0, 35)
# ax.set_ylim(0, 35)
# ax.view_init(elev = 25, azim=135)

# ax.set_xlabel('F')
# ax.set_ylabel('L')
# ax.set_zlabel('Average Runtime (s)')

# plt.title('Average Runtime by F and L')
# plt.show()

# # Runtime heatmap
# rt = df.pivot(index="L", columns="F", values="avg_time")

# fig1, ax1 = plt.subplots()
# im1 = ax1.imshow(rt, origin="lower", aspect="auto", cmap="viridis")
# ax1.set_xlabel("F"); ax1.set_ylabel("L")
# ax1.set_title("Runtime heatmap")
# fig1.colorbar(im1, ax=ax1, label="Avg runtime (s)")

# # OFV heatmap
# ofv = df.pivot(index="L", columns="F", values="avg_ofv")
# fig4, ax4 = plt.subplots()
# im4 = ax4.imshow(ofv, origin="lower", aspect="auto", cmap="coolwarm")
# ax4.set_xlabel("F"); ax4.set_ylabel("L")
# ax4.set_title("OFV heatmap")
# # show ticks aligned with F/L values
# ax4.set_xticks(np.arange(ofv.shape[1])); ax4.set_yticks(np.arange(ofv.shape[0]))
# ax4.set_xticklabels(ofv.columns, rotation=90); ax4.set_yticklabels(ofv.index)
# fig4.colorbar(im4, ax=ax4, label="Avg OFV")

# # show figures (necessary in scripts)
# plt.tight_layout()
# plt.show()


# # Scatter Runtime vs OFV
# fig2, ax2 = plt.subplots()
# sc = ax2.scatter(df["avg_time"], df["avg_ofv"], c=(df["F"] + df["L"]), s=25, alpha=0.7, cmap="plasma")
# ax2.set_xlabel("Avg runtime (s)"); ax2.set_ylabel("Avg OFV")
# ax2.set_title("Runtime vs OFV")
# fig2.colorbar(sc, ax=ax2, label="F+L (window size)")

# # add monolithic markers
# mon_runtime = 29.3   # seconds, at 0.05 optimality gap
# mon_runtime = 29.2   # seconds, at 0.1 optimality gap
# mon_ofv     = 487321336.34 

# # vertical line for monolithic runtime and horizontal line for monolithic OFV
# ax2.axvline(x=mon_runtime, color='black', linestyle='--', linewidth=1, label='Monolithic runtime, 0.05 opt gap')
# ax2.axhline(y=mon_ofv, color='black', linestyle='-.', linewidth=1, label='Monolithic OFV, 0.05 opt gap')

# # annotate the intersection (small boxed label with arrow)
# ax2.annotate(f"Monolithic\nruntime={mon_runtime:.1f}s\nOFV={mon_ofv:.2f}",
#              xy=(mon_runtime, mon_ofv), xycoords='data',
#              xytext=(10,10), textcoords='offset points',
#              fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))

# ax2.legend(loc='best', fontsize=8)

# # optional: log-scaled runtime heatmap in its own figure
# fig3, ax3 = plt.subplots()
# im3 = ax3.imshow(np.log1p(rt), origin="lower", aspect="auto", cmap="magma")
# ax3.set_xlabel("F"); ax3.set_ylabel("L")
# ax3.set_title("Log Runtime heatmap")
# fig3.colorbar(im3, ax=ax3, label="log(1 + avg runtime)")

# show figures (necessary in scripts)
# plt.show()
