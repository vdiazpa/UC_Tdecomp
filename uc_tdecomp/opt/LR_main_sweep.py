#LR_main_sweep.py

from .LR_subp_build import build_LR_subprobs
from ..data.data_extract import load_csv_data, load_rts_data
from ..opt.RH_main import run_RH
from pyomo.environ import *
from time import perf_counter
from multiprocessing import Pool
import numpy as np
import math
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

logging.getLogger('pyomo.core').setLevel(logging.ERROR)

USE_WARMSTART = False

# worker objects (globals)
DATA = None
INDEX_SET = None
TIME_WINDOWS = None
SOLVER_CACHE = {}
MODEL_CACHE = {}
LAST_START = {}

# these are set per run in __main__ and used by get_lagrange_and_g
bats = None
ther_gens = None


# -------------------------- Solver helpers --------------------------

def solve_and_return(m, opt, k, sub_id):
    opt.set_objective(m.Objective)
    results = opt.solve(load_solutions=True, warmstart=USE_WARMSTART)

    inc_obj = float(value(m.Objective))

    # best bound from Gurobi persistent model (valid LB for minimization MIP)
    try:
        gm = opt._solver_model
        # Gurobi uses ObjBound (capital B)
        best_bound = float(gm.ObjBound)
    except Exception as e:
        raise RuntimeError(f"Could not read Gurobi ObjBound for subproblem {sub_id}: {e}")

    LAST_START[sub_id] = _capture_start(m)

    # Store vars as you already do (unchanged), plus incumbent/best_bound
    if 1 in list(m.TimePeriods):
        return_object = {
            'inc_obj': inc_obj,
            'best_bound': best_bound,
            'vars': {
                'PowerGenerated': {(g, t): value(m.PowerGenerated[g, t]) for g in m.ThermalGenerators for t in m.TimePeriods},
                'UnitOn': {(g, t): value(m.UnitOn[g, t]) for g in m.ThermalGenerators for t in m.TimePeriods},
                'UT_Obl_end': {(g, t): value(m.UT_Obl_end[g, t]) for g in m.ThermalGenerators for t in m.Max_t},
                'DT_Obl_end': {(g, t): value(m.DT_Obl_end[g, t]) for g in m.ThermalGenerators for t in m.Max_t},
                'SoC': {(b, t): value(m.SoC[b, t]) for b in m.StorageUnits for t in m.Max_t},
            }
        }
    else:
        return_object = {
            'inc_obj': inc_obj,
            'best_bound': best_bound,
            'vars': {
                'UnitOn_copy': {(g, t): value(m.UnitOn_copy[g, t]) for g in m.ThermalGenerators for t in m.Min_t},
                'PowerGenerated_copy': {(g, t): value(m.PowerGenerated_copy[g, t]) for g in m.ThermalGenerators for t in m.Min_t},
                'UT_Obl_end': {(g, t): value(m.UT_Obl_end[g, t]) for g in m.ThermalGenerators for t in m.Max_t},
                'DT_Obl_end': {(g, t): value(m.DT_Obl_end[g, t]) for g in m.ThermalGenerators for t in m.Max_t},
                'UT_Obl_copy': {(g, t): value(m.UT_Obl_copy[g, t]) for g in m.ThermalGenerators for t in m.Min_t},
                'DT_Obl_copy': {(g, t): value(m.DT_Obl_copy[g, t]) for g in m.ThermalGenerators for t in m.Min_t},
                'PowerGenerated': {(g, t): value(m.PowerGenerated[g, t]) for g in m.ThermalGenerators for t in m.TimePeriods},
                'UnitOn': {(g, t): value(m.UnitOn[g, t]) for g in m.ThermalGenerators for t in m.TimePeriods},
                'SoC_copy': {(b, t): value(m.SoC_copy[b, t]) for b in m.StorageUnits for t in m.Min_t},
                'SoC': {(b, t): value(m.SoC[b, t]) for b in m.StorageUnits for t in m.Max_t},
            }
        }

    return return_object


def get_lagrange_and_g(results_obj, T_windows):
    """
    Returns:
      inc_total : sum of incumbent objectives across subproblems
      lb_total  : sum of best bounds (ObjBound) across subproblems  <-- valid LB for dual plot/table
      g_vec     : coupling mismatch vector (unchanged)
    """
    global bats, ther_gens

    g_vec = {}
    inc_total = 0.0
    lb_total = 0.0
    N = len(results_obj)
    times = [max(T_windows[i]) for i in range(N - 1)]

    for i in range(N):
        inc_total += results_obj[i]['inc_obj']
        lb_total += results_obj[i]['best_bound']

    for g in ther_gens:
        for k in ['UnitOn', 'PowerGenerated', 'DT_Obl', 'UT_Obl']:
            if k == 'UnitOn' or k == 'PowerGenerated':
                for i, t in enumerate(times):
                    g_vec[(g, t, k)] = results_obj[i]['vars'][k][(g, t)] - results_obj[i + 1]['vars'][f'{k}_copy'][(g, t)]
            else:
                for i, t in enumerate(times):
                    g_vec[(g, t, k)] = results_obj[i]['vars'][f'{k}_end'][(g, t)] - results_obj[i + 1]['vars'][f'{k}_copy'][(g, t)]

    for b in bats:
        for i, t in enumerate(times):
            g_vec[(b, t, 'SoC')] = results_obj[i]['vars']['SoC'][(b, t)] - results_obj[i + 1]['vars']['SoC_copy'][(b, t)]

    return inc_total, lb_total, g_vec


def init_globals(data, Time_windows, index_set):
    global DATA, TIME_WINDOWS, INDEX_SET, MODEL_CACHE, SOLVER_CACHE, LAST_START
    DATA = data
    TIME_WINDOWS = Time_windows
    INDEX_SET = index_set
    MODEL_CACHE = {}
    SOLVER_CACHE = {}
    LAST_START = {}


def solver_function(task):
    sub_id, lambda_obj, k = task

    if sub_id not in MODEL_CACHE:
        m = build_LR_subprobs(DATA, TIME_WINDOWS[sub_id], index_set=INDEX_SET)
        opt = SolverFactory('gurobi_persistent')
        opt.options['OutputFlag'] = 0
        opt.options['Presolve'] = 2
        opt.options['Threads'] = 1
        opt.options['MIPGap'] = 0.01
        opt.set_instance(m)
        MODEL_CACHE[sub_id] = m
        SOLVER_CACHE[sub_id] = opt

    m = MODEL_CACHE[sub_id]
    opt = SOLVER_CACHE[sub_id]

    for key, val in lambda_obj.items():
        if key in m.L_index:
            m.L[key] = val

    if USE_WARMSTART and sub_id in LAST_START:
        _apply_start(m, LAST_START[sub_id])

    return solve_and_return(m, opt, k, sub_id)


def _capture_start(m):
    start = {
        'UnitOn': {(g, t): int(round(value(m.UnitOn[g, t])))
                   for g in m.ThermalGenerators for t in m.TimePeriods},

        'UnitStart': {(g, t): int(round(value(m.UnitStart[g, t])))
                      for g in m.ThermalGenerators for t in m.TimePeriods
                      if (g, t) in m.UnitStart},

        'UnitStop': {(g, t): int(round(value(m.UnitStop[g, t])))
                     for g in m.ThermalGenerators for t in m.TimePeriods
                     if (g, t) in m.UnitStop},

        'IsCharging': {(b, t): int(round(value(m.IsCharging[b, t])))
                       for b in m.StorageUnits for t in m.TimePeriods
                       if (b, t) in m.IsCharging},

        'IsDischarging': {(b, t): int(round(value(m.IsDischarging[b, t])))
                          for b in m.StorageUnits for t in m.TimePeriods
                          if (b, t) in m.IsDischarging},

        'PowerGenerated': {(g, t): value(m.PowerGenerated[g, t])
                           for g in m.ThermalGenerators for t in m.TimePeriods},

        'SoC': {(b, t): value(m.SoC[b, t])
                for b in m.StorageUnits for t in m.TimePeriods
                if (b, t) in m.SoC},

        'ChargePower': {(b, t): value(m.ChargePower[b, t])
                        for b in m.StorageUnits for t in m.TimePeriods
                        if (b, t) in m.ChargePower},

        'DischargePower': {(b, t): value(m.DischargePower[b, t])
                           for b in m.StorageUnits for t in m.TimePeriods
                           if (b, t) in m.DischargePower},
    }
    return start


def _apply_start(m, start):
    for (g, t), v in start.get('PowerGenerated', {}).items():
        if (g, t) in m.PowerGenerated:
            m.PowerGenerated[g, t].value = v
    for (b, t), v in start.get('SoC', {}).items():
        if (b, t) in m.SoC:
            m.SoC[b, t].value = v
    for (b, t), v in start.get('ChargePower', {}).items():
        if (b, t) in m.ChargePower:
            m.ChargePower[b, t].value = v
    for (b, t), v in start.get('DischargePower', {}).items():
        if (b, t) in m.DischargePower:
            m.DischargePower[b, t].value = v


def sanitize(x):
    s = f"{x}".replace("_", "m").replace(".", "p")
    return s


def save_run_csv(data, SYS_NAME, T, subn, gamma, gamma_hat,
                 Lag_inc_set, Lag_lb_set, g_norm, Level_vals,
                 out_dir="LR_runs"):
    """
    Saves per-iteration trajectories with BOTH:
      - dual_inc (sum of incumbents)
      - dual_lb  (sum of ObjBounds)  <-- valid LB
      - best_dual (running max of dual_lb)
    """
    if len(data.get("buses")) > 300:
        out_dir = out_dir + f"_DUK_{T}"
    else:
        out_dir = out_dir + f"_RTS_{T}"
    os.makedirs(out_dir, exist_ok=True)

    n = min(len(Lag_inc_set), len(Lag_lb_set), len(Level_vals))
    iters = list(range(n))

    dual_inc = np.asarray(Lag_inc_set[:n], dtype=float)
    dual_lb  = np.asarray(Lag_lb_set[:n], dtype=float)
    best_dual = np.maximum.accumulate(dual_lb)
    level = np.asarray(Level_vals[:n], dtype=float)
    gnorm = np.asarray(g_norm + [np.nan] * (n - len(g_norm)), dtype=float)

    df = pd.DataFrame({
        "iteration": iters,
        "dual_inc": dual_inc,
        "dual_lb": dual_lb,
        "best_dual": best_dual,
        "level": level,
        "g_norm": gnorm
    })

    base = f"{SYS_NAME}_T{T}_W{subn}_g{sanitize(gamma)}_ghat{sanitize(gamma_hat)}"
    csv_path = os.path.join(out_dir, f"lr_{base}.csv")
    meta_path = os.path.join(out_dir, f"lr_{base}.meta.json")

    df.to_csv(csv_path, index=False)
    with open(meta_path, "w") as f:
        json.dump({"system": SYS_NAME, "T": T, "window": subn, "gamma": gamma, "gamma_hat": gamma_hat}, f, indent=2)

    print(f"[saved] {csv_path}")


def save_norms_csv(data, SYS_NAME, T, subn, gamma, g_norm, out_dir="LR_runs"):
    if len(data.get("buses")) > 300:
        out_dir = out_dir + f"_DUK_{T}"
    else:
        out_dir = out_dir + f"_RTS_{T}"
    os.makedirs(out_dir, exist_ok=True)

    iters = np.arange(1, len(g_norm) + 1)
    df = pd.DataFrame({"iteration": iters, "g_norm": np.asarray(g_norm, dtype=float)})

    base = f"{SYS_NAME}_T{T}_W{subn}_g{sanitize(gamma)}"
    path = os.path.join(out_dir, f"norms_{base}.csv")
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def compute_q0_from_rh(data, T, F=24, L=12, rh_opt_gap=0.01,
                       MUT="counter", MDT="counter"):
    rh_time, ub, fixed_sol = run_RH(
        data=data, F=F, L=L, T=T, RH_opt_gap=rh_opt_gap,
        verbose=False, write_csv=False, s_tee=False, benchmark=False,
        seed=None, MUT=MUT, MDT=MDT
    )

    if ub is None or (isinstance(ub, float) and math.isnan(ub)):
        raise RuntimeError("RH-based q0 failed: run_RH returned ub=None/NaN")

    print(f"[q0] Using RH UB as q0: {ub:.2f}  (RH time={rh_time:.2f}s, F={F}, L={L}, gap={rh_opt_gap})")
    return ub


# -------------------------- MAIN --------------------------

if __name__ == "__main__":

    # Run order: RTS first, then DUK; and T increasing 72 -> 168 -> 336
    TIMES = [72, 168, 336]
    SYSTEMS = [
        #("RTS", load_rts_data),
        ("DUK", load_csv_data),
    ]

    #subn = 48
    max_iters = 30
    LAMBDA_CONFIGS = [(0.2, 1), (0.4, 1), (0.6, 1)]

    WINDOWS = [48]
    for SYS_NAME, LOADER in SYSTEMS:
        for T in TIMES:
            for subn in WINDOWS:
                print("\n" + "=" * 80)
                print(f"RUNNING LR: system={SYS_NAME}, T={T}")
                print("=" * 80)

                # ---------------- Load data ----------------
                data = LOADER(T)
                bats = data["bats"]
                ther_gens = data["ther_gens"]
                num_periods = len(data["periods"])

                # ---------------- Time windows ----------------
                Time_windows = [list(range(i, min(i + subn, num_periods + 1)))
                                for i in range(1, num_periods + 1, subn)]

                print("num_periods:", num_periods,
                    "\nWindow size:", subn,
                    "\n# of windows (subproblems):", len(Time_windows))

                # ---------------- Initial lambdas / index set ----------------
                pow_pnlty, commit_pnlty, UT_pnlty, DT_pnlty, soc_pnlty = 35, 30, 40, 32, 25
                index_set = set()
                l = {}

                for i in range(subn, num_periods, subn):
                    for var in ['UnitOn', 'PowerGenerated', 'DT_Obl', 'UT_Obl']:
                        for g in ther_gens:
                            index_set.add((g, i, var))
                    for b in bats:
                        index_set.add((b, i, 'SoC'))

                for key in index_set:
                    if 'UnitOn' in key:
                        l[key] = commit_pnlty
                    elif 'PowerGenerated' in key:
                        l[key] = pow_pnlty
                    elif 'DT_Obl' in key:
                        l[key] = DT_pnlty
                    elif 'SoC' in key:
                        l[key] = soc_pnlty
                    else:
                        l[key] = UT_pnlty

                l0 = dict(l)

                # ---------------- Pool init (per run) ----------------
                num_cores = 12
                num_winds = len(Time_windows)
                t_all_start = perf_counter()

                pool = Pool(
                    processes=min(num_winds, num_cores),
                    initializer=init_globals,
                    initargs=(data, Time_windows, index_set)
                )

                # Initial solve
                tasks0 = [(i, l0, 0) for i in range(len(Time_windows))]
                results0 = pool.map(solver_function, tasks0)
                Lag0_inc, Lag0_lb, g0 = get_lagrange_and_g(results0, Time_windows)

                # RH UB for q0
                q0 = compute_q0_from_rh(data, T=T, F=24, L=12, rh_opt_gap=0.01, MUT="counter", MDT="counter")

                t_after_init = perf_counter()
                print(f"\nInitial build and solve time: {t_after_init - t_all_start:.3f} secs")
                print("q0 (UB):", f"{q0:.2f}")
                print("Initial dual (inc sum):", f"{Lag0_inc:.2f}")
                print("Initial dual LB (ObjBound sum):", f"{Lag0_lb:.2f}")

                # ---------------- Main LR loop over gamma ----------------
                runs = []
                for (gamma, gamma_hat) in LAMBDA_CONFIGS:

                    run_start = perf_counter()
                    print(f"\n\nStarting new LR run with gamma={gamma}, gamma_hat={gamma_hat}\n")

                    y_LB, y_UB = -np.inf, np.inf
                    PSVD = ConcreteModel()
                    PSVD.y = Var(index_set, within=Reals, bounds=(y_LB, y_UB))
                    PSVD.Inf_det = ConstraintList(doc='inf_detect')
                    opt = SolverFactory('gurobi')

                    l = dict(l0)

                    # Use LB track for certification, incumbent track for stepsize stability
                    L_best = Lag0_lb
                    Lag_inc = Lag0_inc
                    Lag_lb = Lag0_lb
                    q = q0
                    g = g0
                    UB = q0

                    g_length, iter_times = [], []

                    Lag_inc_set = [Lag0_inc]
                    Lag_lb_set = [Lag0_lb]
                    Level_vals = [q0]

                    eta, j, k = 0, 0, 0
                    k_j = [0]

                    while k < max_iters:
                        t0 = perf_counter()

                        if k > 0:
                            tasks = [(i, l, k) for i in range(len(Time_windows))]
                            results = pool.map(solver_function, tasks)

                            Lag_inc, Lag_lb, g = get_lagrange_and_g(results, Time_windows)

                            L_best = max(L_best, Lag_lb)

                            Lag_inc_set.append(Lag_inc)
                            Lag_lb_set.append(Lag_lb)
                        else:
                            Lag_inc, Lag_lb, g = Lag0_inc, Lag0_lb, g0

                        g_norm = np.linalg.norm(np.array(list(g.values())), ord=2)
                        diff = q - Lag_inc
                        s_k = gamma * (diff / (g_norm ** 2 + 1e-10))

                        g_length.append(g_norm)

                        PSVD.Inf_det.add(
                            sum(g[r] * PSVD.y[r] for r in index_set) >=
                            sum(g[r] * l[r] for r in index_set) + (1 / gamma_hat) * s_k * g_norm ** 2
                        )
                        PSVD_results = opt.solve(PSVD, tee=False)

                        # update multipliers
                        for r in g:
                            l[r] = l[r] + g[r] * s_k

                        if k == max_iters - 1:
                            with open(f"l_final_{SYS_NAME}_T{T}_g{sanitize(gamma)}.csv", "w", newline="") as f:
                                w = csv.writer(f)
                                w.writerow(["g", "t", "var", "multiplier"])
                                for (gg, tt, var), val in sorted(l.items()):
                                    w.writerow([gg, tt, var, val])
                            print("Saved final multipliers.")

                        if PSVD_results.solver.termination_condition != TerminationCondition.optimal:
                            q = (gamma / gamma_hat) * q + (1 - gamma / gamma_hat) * max(Lag_lb_set[k_j[-1]:k_j[-1] + eta + 1])
                            PSVD.Inf_det.clear()
                            Level_vals.append(q)
                            k_j.append(k)
                            eta = 0
                            j += 1
                        else:
                            eta += 1
                            Level_vals.append(q)

                        t1 = perf_counter()
                        iter_times.append(t1 - t0)

                        gap_pct = 100.0 * (UB - L_best) / (abs(UB) if abs(UB) > 1e-12 else 1.0)
                        print(f"Iter {k} took {t1 - t0:.2f}s; dual_inc={Lag_inc:.2f}, dual_lb={Lag_lb:.2f}, "
                            f"best_lb={L_best:.2f}, level={q:.2f}, gap={gap_pct:.2f}%")

                        k += 1

                    LR_runtime = perf_counter() - run_start
                    print(f"Total LR time = {LR_runtime:.2f}s, avg iter time = {np.mean(iter_times):.2f}s")

                    # One extra evaluation with final multipliers (append both)
                    results_final = pool.map(solver_function, [(i, l, max_iters) for i in range(len(Time_windows))])
                    Lag_final_inc, Lag_final_lb, _ = get_lagrange_and_g(results_final, Time_windows)

                    Lag_inc_set.append(Lag_final_inc)
                    Lag_lb_set.append(Lag_final_lb)
                    Level_vals.append(q)

                    runs.append({
                        'gamma': gamma,
                        'gamma_hat': gamma_hat,
                        'Lag_inc_set': Lag_inc_set,
                        'Lag_lb_set': Lag_lb_set,
                        'Level_vals': Level_vals,
                        'Runtime': LR_runtime,
                        'g_norm': g_length
                    })

                    # Save trajectories with both columns
                    save_run_csv(
                        data, SYS_NAME, T, subn, gamma, gamma_hat,
                        Lag_inc_set, Lag_lb_set, g_length, Level_vals
                    )
                    save_norms_csv(data, SYS_NAME, T, subn, gamma, g_length)

                print("\n############### Summary #################")
                for r in runs:
                    print(f"system={SYS_NAME}, T={T}, gamma={r['gamma']}, "
                        f"iters={len(r['Lag_lb_set'])-1}, runtime={r['Runtime']:.2f}s, "
                        f"best_dual_lb={max(r['Lag_lb_set']):.2f}")

                pool.close()
                pool.join()