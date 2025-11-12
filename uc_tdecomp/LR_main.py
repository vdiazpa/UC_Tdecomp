
from uc_tdecomp.LR_subp_build import build_LR_subprobs
from uc_tdecomp.data_extract import load_uc_data, load_csv_data
from pyomo.environ import *
from time import perf_counter
from multiprocessing import Pool
import numpy as np
import math
import csv
import multiprocessing as mp

import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

USE_WARMSTART = False
#WARMSTART_AFTER = 10

# worker objects (globals)
 # Module level names that will exist in every worker process
DATA = None
INDEX_SET    = None
TIME_WINDOWS = None
SOLVER_CACHE = {}
MODEL_CACHE  = {}
LAST_START   = {}

#Define functions for main loop 
def solve_and_return(m, opt, k, sub_id): #, use_warm):
    
    opt.set_objective(m.Objective)
    
    results = opt.solve(load_solutions = True, warmstart = USE_WARMSTART)  # soles without rebuilding

    LAST_START[sub_id] = _capture_start(m)

    if 1 in list(m.TimePeriods):
                return_object = {  'ofv': value(m.Objective), 
            'vars': {
                'PowerGenerated':  {(g,t): value(m.PowerGenerated[g,t])  for g in m.ThermalGenerators for t in m.TimePeriods },
                'UnitOn':          {(g,t): value(m.UnitOn[g,t])          for g in m.ThermalGenerators for t in m.TimePeriods}, 
                'UT_Obl_end':      {(g,t): value(m.UT_Obl_end[g,t])      for g in m.ThermalGenerators for t in m.Max_t},
                'DT_Obl_end':      {(g,t): value(m.DT_Obl_end[g,t])      for g in m.ThermalGenerators for t in m.Max_t},
                'SoC':             {(b,t): value(m.SoC[b,t])             for b in m.StorageUnits      for t in m.Max_t}}}
    else: 
        return_object = {  'ofv': value(m.Objective), 
            'vars': {
                'UnitOn_copy':         {(g,t): value(m.UnitOn_copy[g,t])         for g in m.ThermalGenerators for t in m.Min_t}, 
                'PowerGenerated_copy': {(g,t): value(m.PowerGenerated_copy[g,t]) for g in m.ThermalGenerators for t in m.Min_t},
                'UT_Obl_end':          {(g,t): value(m.UT_Obl_end[g,t])          for g in m.ThermalGenerators for t in m.Max_t},
                'DT_Obl_end':          {(g,t): value(m.DT_Obl_end[g,t])          for g in m.ThermalGenerators for t in m.Max_t},
                'UT_Obl_copy':         {(g,t): value(m.UT_Obl_copy[g,t])         for g in m.ThermalGenerators for t in m.Min_t},
                'DT_Obl_copy':         {(g,t): value(m.DT_Obl_copy[g,t])         for g in m.ThermalGenerators for t in m.Min_t},
                'PowerGenerated':      {(g,t): value(m.PowerGenerated[g,t])      for g in m.ThermalGenerators for t in m.TimePeriods },
                'UnitOn':              {(g,t): value(m.UnitOn[g,t])              for g in m.ThermalGenerators for t in m.TimePeriods },
                'SoC_copy':            {(b,t): value(m.SoC_copy[b,t])            for b in m.StorageUnits      for t in m.Min_t},
                'SoC':                 {(b,t): value(m.SoC[b,t])                 for b in m.StorageUnits      for t in m.Max_t}}}
        
    # if k ==1:
    #     m.write(f'm.{str(s_e)}_{k}.lp', io_options = {'symbolic_solver_labels': True})
    #     csv_filename = f"solution_{str(s_e)}_{k}.csv"
    #     with open(csv_filename, mode='w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(['Var', 'g', 't', 'Value'])

    #         for varname, index_dict in return_object['vars'].items():
    #             for (g,t), val in index_dict.items():
    #                 writer.writerow([varname, g, t, val])

    return return_object

def get_lagrange_and_g(results_obj, T_windows):
    g_vec      = {}
    ofv_total  = 0
    N          = len(results_obj)
    times      = [ max(T_windows[i]) for i in range(N-1) ]  

    for i in range(N):
        ofv_total +=results_obj[i]['ofv'] # sums all objective functions
        
    for g in ther_gens: 
        for k in ['UnitOn', 'PowerGenerated', 'DT_Obl', 'UT_Obl']:
            if k == 'UnitOn' or k == 'PowerGenerated':
                for i, t in enumerate(times):  
                    g_vec[(g, t, k)] = results_obj[i]['vars'][k][(g,t)] - results_obj[i+1]['vars'][f'{k}_copy'][(g,t)]
            else:
                for i, t in enumerate(times):
                    g_vec[(g, t, k)] = results_obj[i]['vars'][f'{k}_end'][(g,t)] - results_obj[i+1]['vars'][f'{k}_copy'][(g,t)]
                    
    for b in bats:
        for i, t in enumerate(times):
            g_vec[(b, t, 'SoC')] = results_obj[i]['vars']['SoC'][(b,t)] - results_obj[i+1]['vars']['SoC_copy'][(b,t)]

    return ofv_total, g_vec

def init_globals(data, Time_windows, index_set):
    global DATA, TIME_WINDOWS, INDEX_SET, MODEL_CACHE, SOLVER_CACHE, LAST_START
    DATA         = data
    TIME_WINDOWS = Time_windows
    INDEX_SET    = index_set
    MODEL_CACHE  = {}
    SOLVER_CACHE = {}
    LAST_START   = {}
    
def solver_function(task):
    sub_id, lambda_obj, k = task

    # Build & cache model+solver for this subproblem the first time we see it
    if sub_id not in MODEL_CACHE:
        m = build_LR_subprobs(DATA, TIME_WINDOWS[sub_id], index_set=INDEX_SET)
        opt = SolverFactory('gurobi_persistent')
        opt.options['OutputFlag'] = 0
        opt.options['Presolve'] = 2
        opt.options['Threads'] = 1
        opt.options['MIPGap'] = 0.05
        opt.set_instance(m)
        MODEL_CACHE[sub_id]  = m
        SOLVER_CACHE[sub_id] = opt

    m   = MODEL_CACHE[sub_id]
    opt = SOLVER_CACHE[sub_id]

    # Update Lagrange multipliers in-place
    for key, val in lambda_obj.items():
        if key in m.L_index:
            m.L[key] = val
          
    # use_warm = (USE_WARMSTART and k >= WARMSTART_AFTER) 
    # if use_warm and sub_id in LAST_START:
    #     _apply_start(m, LAST_START[sub_id])
     
    if USE_WARMSTART and sub_id in LAST_START:
        _apply_start(m, LAST_START[sub_id])
    
    return solve_and_return(m, opt, k, sub_id)

    # return solve_and_return(m, opt, k, sub_id, use_warm)

def _capture_start(m):
    start = {
        # binaries
        'UnitOn':         {(g,t): int(round(value(m.UnitOn[g,t])))
                           for g in m.ThermalGenerators for t in m.TimePeriods},
        'UnitStart':      {(g,t): int(round(value(m.UnitStart[g,t])))
                           for g in m.ThermalGenerators for t in m.TimePeriods
                           if (g,t) in m.UnitStart},   # guard if your model variant changes
        'UnitStop':       {(g,t): int(round(value(m.UnitStop[g,t])))
                           for g in m.ThermalGenerators for t in m.TimePeriods
                           if (g,t) in m.UnitStop},
        'IsCharging':     {(b,t): int(round(value(m.IsCharging[b,t])))
                           for b in m.StorageUnits for t in m.TimePeriods
                           if (b,t) in m.IsCharging},
        'IsDischarging':  {(b,t): int(round(value(m.IsDischarging[b,t])))
                           for b in m.StorageUnits for t in m.TimePeriods
                           if (b,t) in m.IsDischarging},

        # key continuous
        'PowerGenerated': {(g,t): value(m.PowerGenerated[g,t])
                           for g in m.ThermalGenerators for t in m.TimePeriods},
        'SoC':            {(b,t): value(m.SoC[b,t])
                           for b in m.StorageUnits for t in m.TimePeriods
                           if (b,t) in m.SoC},
        'ChargePower':    {(b,t): value(m.ChargePower[b,t])
                           for b in m.StorageUnits for t in m.TimePeriods
                           if (b,t) in m.ChargePower},
        'DischargePower': {(b,t): value(m.DischargePower[b,t])
                           for b in m.StorageUnits for t in m.TimePeriods
                           if (b,t) in m.DischargePower}}
    return start

def _apply_start(m, start):
    """Assign Var.value; persistent solver reads these on set_warm_start()."""
    # binaries
    # for (g,t), v in start.get('UnitOn', {}).items():
    #     if (g,t) in m.UnitOn: m.UnitOn[g,t].value = v
    # for (g,t), v in start.get('UnitStart', {}).items():
    #     if (g,t) in m.UnitStart: m.UnitStart[g,t].value = v
    # for (g,t), v in start.get('UnitStop', {}).items():
    #     if (g,t) in m.UnitStop: m.UnitStop[g,t].value = v
    # for (b,t), v in start.get('IsCharging', {}).items():
    #     if (b,t) in m.IsCharging: m.IsCharging[b,t].value = v
    # for (b,t), v in start.get('IsDischarging', {}).items():
    #     if (b,t) in m.IsDischarging: m.IsDischarging[b,t].value = v

    # continuous
    for (g,t), v in start.get('PowerGenerated', {}).items():
        if (g,t) in m.PowerGenerated: m.PowerGenerated[g,t].value = v
    for (b,t), v in start.get('SoC', {}).items():
        if (b,t) in m.SoC: m.SoC[b,t].value = v
    for (b,t), v in start.get('ChargePower', {}).items():
        if (b,t) in m.ChargePower: m.ChargePower[b,t].value = v
    for (b,t), v in start.get('DischargePower', {}).items():
        if (b,t) in m.DischargePower: m.DischargePower[b,t].value = v


if __name__ == "__main__":

    #======================================================================= Load Data =====

    T = 168
    #file_path  = "examples/unit_commitment/RTS_GMLC_zonal_noreserves.json"
    #data        = load_uc_data(file_path)
    data        = load_csv_data(T)
    bats        = data["bats"]
    ther_gens   = data["ther_gens"]
    num_periods = len(data["periods"])
    
    #============================================================= Set Time Windows ===
    
    subn         = 48                             # number of periods in subproblem
    num_sh       = math.ceil(num_periods/subn)     # number of subproblems
    Time_windows = [list(range(i, min(i+subn, num_periods+1))) for i in range(1, num_periods+1, subn)]
    
    print("num_periods:", num_periods, "\n", "Window size:", subn , "\n", "# of windows (subproblems):", len(Time_windows) )

    #============================================================ Initial Lambdas / Index Set ====
    
    pow_pnlty, commit_pnlty, UT_pnlty, DT_pnlty, soc_pnlty = 35, 30, 40, 32, 25
    index_set = set()
    l = {}
    
    for i in range(subn, num_periods, subn):
        for var in ['UnitOn', 'PowerGenerated', 'DT_Obl', 'UT_Obl']:
            for g in ther_gens: 
                    index_set.add((g, i, var))
        for b in bats:
            index_set.add((b,i,'SoC'))
            
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

    l0 = dict(l)  # initial lambda values
        
    #==========================================================# Initialize Pool Process #=======
    num_cores = 12
    num_winds = len(Time_windows)
    t_all_start = perf_counter()
    pool     = Pool(processes=min(num_winds, num_cores), initializer = init_globals, initargs = (data, Time_windows, index_set))
    tasks0   = [(i, l, 0) for i in range(len(Time_windows))]
    results0 = pool.map(solver_function, tasks0)            # solves and stores
    Lag0, g0 = get_lagrange_and_g(results0, Time_windows)
    
    if T == 168: 
        q0 =  2822785549.96 #w/ storage
    elif T ==336: 
        q0 = 5964866534.55
    else: 
        q0 =  1175106609.19  #w/ storage
        
    t_after_init = perf_counter()
    print(f"\nInitial build and solve time: {t_after_init - t_all_start:.3f} secs", "\nqbar is: ", f"{q0:.2f}", '\n', "Dual Value is", f"{Lag0:.2f}")

    #==========================================================# Initialize Main Loop #==========    
    max_iters    = 30
    LAMBDA_CONFIGS = [(0.2, 1), (0.4, 1), (0.6, 1)]

    runs = []
    for (gamma, gamma_hat) in LAMBDA_CONFIGS:
        
        run_start = perf_counter()
        print(f"\n\nStarting new LR run with gamma={gamma}, gamma_hat={gamma_hat}\n")

        y_LB, y_UB   = -np.inf, np.inf               # Initialize PSVD model
        PSVD         = ConcreteModel() 
        PSVD.y       = Var(index_set, within = Reals, bounds = (y_LB, y_UB))
        PSVD.Inf_det = ConstraintList(doc = 'inf_detect')
        opt          = SolverFactory('gurobi')
  
        l = dict(l0)                                 
        L_best = Lag0
        Lag = Lag0
        q   = q0
        g   = g0
        UB  = q0 
    
        g_length, g_history, iter_times = [], [], []    
        gap_pct = 100.0 * (UB - L_best) / abs(UB)
        eta, j, k  = 0, 0, 0
        Best_lag   = [Lag0]
        Lag_set    = [Lag0]
        Level_vals = [q0]
        g_set      = [g0]
        k_j        = [0]              
    
    #=====================================================================  Main Loop ======
        while k < max_iters:
            t0 = perf_counter()
            if k > 0:
                tasks   = [(i, l, k) for i in range(len(Time_windows))]
                results = pool.map(solver_function, tasks)                 # solves and stores
                Lag, g  = get_lagrange_and_g(results, Time_windows)
                L_best  = max(L_best, Lag)
                gap_pct = 100.0 * (UB - L_best) / abs(UB)
                Best_lag.append(L_best)
                Lag_set.append(Lag); g_set.append(g)
            else:
                Lag, g = Lag0, g0

            g_norm = np.linalg.norm(np.array(list(g.values())), ord = 2)
            diff   = q - Lag
            s_k    = gamma * (diff / (g_norm**2 + 1e-10)) #stepsize
            g_history.append(g)
            g_length.append(g_norm)
            print("\ng length is:", f"{g_norm:.4f}", "\nStepsize is:", f"{s_k:.2f}")# , "\nG is:", g)

            PSVD.Inf_det.add( sum( g[r] * PSVD.y[r] for r in index_set ) >= sum( g[r] * l[r] for r in index_set ) + (1/gamma_hat) * s_k * g_norm**2 )
            PSVD_results = opt.solve(PSVD, tee=False)

            g_scld = { r : g[r] * s_k        for r in g }
            l      = { r : l[r] + g_scld[r]  for r in g } 
        
            if k == max_iters - 1:
                with open("l_final.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["g", "t", "var", "multiplier"])
                    for (g,t,var), val in sorted(l.items()):
                        w.writerow([g, t, var, val])
                print('Saved final multipliers to l_final.csv')
            
            print("PSVD results:", PSVD_results.solver.termination_condition )
            
            if PSVD_results.solver.termination_condition != TerminationCondition.optimal:
                print("Infeasible PSVD solution detected. Adjusting step size.")
                q   = (gamma/gamma_hat) * q + (1 - gamma/gamma_hat) * max(Lag_set[ k_j[-1] : k_j[-1]+eta+1 ])
                print("New q value is:", f"{q:.2f}")
                PSVD.Inf_det.clear()  # clear the constraint list 
                Level_vals.append(q)
                k_j.append(k)
                eta = 0
                j+=1

            else:
                eta += 1
                Level_vals.append(q)

            t1 = perf_counter()
            iter_times.append(t1 - t0)
            print(f"Iter {k} took {t1 - t0:.2f}s;  current dual={Lag:.2f}, level={q:.2f}, gap={gap_pct:.2f}%")
            k+=1
            
        t_end = perf_counter()
        LR_runtime = t_end - run_start
        lbest = np.maximum.accumulate(np.array(Lag_set))
        print(f"Total LR time = {LR_runtime:.2f}s, "f"avg iter time = {np.mean(iter_times):.2f}s")
        print("Best Lagrangian values over iterations:", lbest)

        # one extra evaluation to show the dual with the final multipliers
        results_final = pool.map(solver_function, [(i, l, max_iters) for i in range(len(Time_windows))])
        Lag_final, _ = get_lagrange_and_g(results_final, Time_windows)

        Lag_set.append(Lag_final)   
        Level_vals.append(q)       
        
        runs.append({ 'gamma': gamma, 'gamma_hat': gamma_hat, 'Lag_set': Lag_set, 'Level_vals': Level_vals, 'Runtime': LR_runtime, 'g_norm': g_length })

    print("\n############### Summary of runs #################")
    for r in runs:
        print(f"gamma={r['gamma']}, gamma_hat={r['gamma_hat']}, "
            f"iters={len(r['Lag_set'])-1}, runtime={r['Runtime']:.2f}s, "
            f"best_dual={max(r['Lag_set']):.2f}")

    pool.close()
    pool.join()
    
    import numpy as np
    import matplotlib.pyplot as plt

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure()

    for i, r in enumerate(runs):
        dual  = np.array(r['Lag_set'])
        lbest = np.maximum.accumulate(dual)
        iters = np.arange(len(dual))
        c = colors[i % len(colors)]

        plt.plot(iters, dual, color=c, alpha=0.20, linewidth=1.2, label="_nolegend_")

        # visible running max with legend
        plt.plot(iters, lbest, color=c, linewidth=2.4,label=f"γ={r['gamma']}, ĥγ={r['gamma_hat']}")

    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title(f'Dual convergence across Polyak settings (T={T}, window={subn})')
    plt.legend(ncol=1,fontsize=9, loc='lower right',   frameon=True,fancybox=True,framealpha=0.9,borderpad=0.6,handlelength=2.0 )
    plt.tight_layout()
    plt.savefig(f'lr_convergence_overlay_T{T}_W{subn}.svg', bbox_inches='tight')
    plt.show()
    plt.figure()
    

    for i, r in enumerate(runs):
        norms = np.asarray(r['g_norm'])
        iters = np.arange(1, len(norms)+1)
        c = colors[i % len(colors)]
        plt.plot(iters, norms, color=c, linewidth=2.0,
                label=f"γ={r['gamma']}, ĥγ={r['gamma_hat']}")

    plt.xlabel('Iteration')
    plt.ylabel(r'$\|g\|_2$')
    plt.title(r'Subgradient norm over iterations: $\|g\|_2$')

    plt.legend(
        ncol=1,
        fontsize=9,
        loc='lower right',
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        borderpad=0.6,
        handlelength=2.0
    )

    plt.tight_layout()
    plt.savefig(f'lr_norms_overlay_T{T}_W{subn}.svg', bbox_inches='tight')
    plt.show()

    
    
    
    
    # #Plot Results
    # import numpy as np
    # import matplotlib.pyplot as plt

    # dual  = np.array(Lag_set)
    # lbest = np.maximum.accumulate(dual)
    # iters = np.arange(len(dual))

    # plt.figure()
    # plt.plot(iters, lbest, label='Best Lagrangian (running max)', color='C1', linestyle='--')
    # plt.plot(iters, dual,  label='Dual at iteration k', color='C0', linestyle='-')

    # level = np.array(Level_vals[:len(dual)])
    # plt.plot(iters, level, label='Level Value', color='C2')

    # plt.xlabel('Iteration'); plt.ylabel('Value')
    # plt.title('Dual and Best Lagrangian Lower Bound Over Iterations')
    # plt.legend(); plt.tight_layout()
    # plt.savefig('dual_vs_lbest_gamma={gamma}_hat={gamma_hat}_bounds=({y_LB},{y_UB})_T={T}.svg')                 # or: plt.savefig('dual_vs_lbest.svg', format='svg')
    # plt.show()


    # dual  = np.array(Lag_set)
    # level = np.array(Level_vals)
    # n     = min(len(dual), len(level))
    # dual  = dual[:n]
    # level = level[:n]
    # iters = np.arange(n)
    # plt.figure()
    # plt.plot(iters, dual, label='Dual (Lagrange) Values')
    # plt.plot(iters, level, label='Level Values')
    # plt.xlabel('Iteration')
    # plt.ylabel('Values')
    # plt.title(f'Dual and Level Values Over Iterations')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f'dual_and_level_values_y={abs(y_UB)}.png')
    # plt.show()

    # g_l  = np.array(g_length)
    # n     = len(g_length)
    # levg_lel = level[:n]
    # iters = np.arange(n)
    # plt.figure()
    # plt.plot(iters, g_l, label='Length of g')
    # plt.xlabel('Iteration')
    # plt.ylabel('Values')
    # plt.title(f'Norm of g')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('g_length.png')
    # plt.show()

    # for key in g_history[0].keys():
    #     if 'UnitOn' in key:
    #         plt.title('Absolute Value of g for Commitment Var Across Iterations')
    #         plt.plot(range(len(g_history)), [abs(g[key]) for g in g_history], marker='o', linestyle='-', label=key)
    # plt.xlabel('Iteration')
    # plt.ylabel('g values')
    # plt.show()

    # for key in g_history[0].keys():
    #     if 'PowerGenerated' in key:
    #         plt.title('Absolute Value of g for Power Generated Var Across Iterations')
    #         plt.plot(range(len(g_history)), [abs(g[key]) for g in g_history], marker='o', linestyle='-', label=key)

    # plt.xlabel('Iteration')
    # plt.ylabel('g values')
    # plt.show()
