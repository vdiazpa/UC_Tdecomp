
from uc_tdecomp.LR_subp_build_t import build_subprobs_t
from uc_tdecomp.data_extract import load_uc_data, load_csv_data
from pyomo.environ import *
from time import perf_counter
from multiprocessing import Pool
import numpy as np
import math
import csv

# worker objects (globals)
 # Module level names that will exist in every worker process
DATA = None
INDEX_SET    = None
TIME_WINDOWS = None
SOLVER_CACHE = {}
MODEL_CACHE  = {}
LAST_START   = {}

#Define functions for main loop 
def solve_and_return(m, opt, k):
    
    opt.set_objective(m.Objective)

    results = opt.solve(load_solutions = True)  # soles without rebuilding

    if 1 in list(m.TimePeriods):
                return_object = {  'ofv': value(m.Objective), 
            'vars': {
                'PowerGenerated':  {(g,t): value(m.PowerGenerated[g,t])  for g in m.ThermalGenerators for t in m.TimePeriods },
                'UnitOn':          {(g,t): value(m.UnitOn[g,t])          for g in m.ThermalGenerators for t in m.TimePeriods}, 
                'UT_Obl_end':      {(g,t): value(m.UT_Obl_end[g,t])      for g in m.ThermalGenerators for t in m.Max_t},
                'DT_Obl_end':      {(g,t): value(m.DT_Obl_end[g,t])      for g in m.ThermalGenerators for t in m.Max_t}}}
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
                'UnitOn':              {(g,t): value(m.UnitOn[g,t])              for g in m.ThermalGenerators for t in m.TimePeriods }}}
        
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
        m = build_subprobs_t(DATA, TIME_WINDOWS[sub_id], index_set=INDEX_SET)
        opt = SolverFactory('gurobi_persistent')
        opt.options['OutputFlag'] = 0
        opt.options['Presolve'] = 2
        opt.options['MIPGap'] = 0.2
        opt.set_instance(m)
        MODEL_CACHE[sub_id]  = m
        SOLVER_CACHE[sub_id] = opt

    m   = MODEL_CACHE[sub_id]
    opt = SOLVER_CACHE[sub_id]

    # Update Lagrange multipliers in-place (no re-construction)
    for key, val in lambda_obj.items():
        if key in m.L_index:
            m.L[key] = val

    # (Optional) warm start:
    # If you saved last solution for this sub_id, assign to m.Var[...] .value = ...
    # then inform the persistent solver to use a warm start:
    # opt.set_warm_start()  # for pyomo.gurobi_persistent, this reads Var.value as MIP start

    return solve_and_return(m, opt, k)

if __name__ == "__main__":

    #======================================================================= Load Data =====
    T = 72
    #file_path  = "examples/unit_commitment/RTS_GMLC_zonal_noreserves.json"
    #data        = load_uc_data(file_path)
    data        = load_csv_data(T)
    ther_gens   = data["ther_gens"]
    num_periods = len(data["periods"])
    
    #============================================================= Set Time Windows ===
    
    subn         = 24                             # number of periods in subproblem
    num_sh       = math.ceil(num_periods/subn)     # number of subproblems
    Time_windows = [list(range(i, min(i+subn, num_periods+1))) for i in range(1, num_periods+1, subn)]
    
    print("num_periods:", num_periods, "\n", "Window size:", subn , "\n", "# of windows (subproblems):", len(Time_windows) )

    #============================================================ SET INIT LAMBDA VALUES ====
    
    pow_pnlty, commit_pnlty, UT_pnlty, DT_pnlty = 35, 30, 40, 32
    index_set = set()
    l = {}
    
    for i in range(subn, num_periods, subn):
        for var in ['UnitOn', 'PowerGenerated', 'DT_Obl', 'UT_Obl']:
            for g in ther_gens: 
                    index_set.add((g, i, var))
   
    for key in index_set:
        if 'UnitOn' in key:
            l[key] = commit_pnlty
        elif 'PowerGenerated' in key:
            l[key] = pow_pnlty
        elif 'DT_Obl' in key:
            l[key] = DT_pnlty
        else:
            l[key] = UT_pnlty

    #=========================================================== Initialize Pool Process ======

    t_all_start = perf_counter()
    pool = Pool(processes=len(Time_windows), initializer = init_globals, initargs = (data, Time_windows, index_set))
    
    tasks0   = [(i, l, 0) for i in range(len(Time_windows))]
    results0 = pool.map(solver_function, tasks0)            # solves and stores
    Lag0, g0 = get_lagrange_and_g(results0, Time_windows)
    
    #q0 = Lag0 + 0.1* (abs(Lag0) + 1.0)
    if T == 168: 
        q0 = 2471099193.65292
    else: 
        q0 = 1021672097.99475
        
    t_after_init = perf_counter()
    print(f"\nInitial build and solve time: {t_after_init - t_all_start:.3f} secs")
    print("qbar is: ", f"{q0:.2f}", '\n', "Dual Value is", f"{Lag0:.2f}")

    #===================================================================== Initialize Main Loop ======
    # gamma      = 1/num_sh   
    max_iters    = 30
    gamma        = 0.5
    gamma_hat    = 1.5

    y_LB, y_UB   = -np.inf, np.inf
    lam_len      = range(len(g0))         # Initialize PSVD model
    PSVD         = ConcreteModel() 
    PSVD.y       = Var(index_set, within = Reals, bounds = (y_LB, y_UB))
    PSVD.Inf_det = ConstraintList(doc = 'inf_detect')
    opt          = SolverFactory('gurobi')

    L_best = Lag0
    Lag = Lag0
    q   = q0
    g   = g0
    
    g_length, g_history, iter_times = [], [], []    
    Best_lag   = [Lag0]
    Lag_set    = [Lag0]
    Level_vals = [q0]
    g_set      = [g0]
    k_j        = [0]              
    UB = q0      
    gap_pct = 100.0 * (UB - L_best) / abs(UB)

    eta, j, k  = 0, 0, 0
    
    #=====================================================================  Main Loop ======
    while k < max_iters:
        t0 = perf_counter()
        if k > 0:
            tasks   = [(i, l, k) for i in range(len(Time_windows))]
            results = pool.map(solver_function, tasks)            # solves and stores
            Lag, g  = get_lagrange_and_g(results, Time_windows)
            L_best  = max(L_best, Lag)
            Best_lag.append(L_best)
            Lag_set.append(Lag); g_set.append(g)
            gap_pct = 100.0 * (UB - L_best) / abs(UB)
        else:
            Lag, g = Lag0, g0

        g_norm = np.linalg.norm(np.array(list(g.values())), ord = 2)
        diff   = q - Lag
        s_k    = gamma * (diff / (g_norm**2 + 1e-10)) #stepsize

        g_history.append(g)
        g_length.append(g_norm)
        print("\ng length is:", f"{g_norm:.4f}", "\nStepsize is:", f"{s_k:.2f}")# , "\nG is:", g)

        PSVD.Inf_det.add( sum( g[r] * PSVD.y[r] for r in index_set ) >= sum( g[r] * l[r] for r in index_set ) + (1/gamma_hat) * s_k * g_norm**2 )
        #PSVD.inf_detect_cut.pprint()

        PSVD_results = opt.solve(PSVD, tee=False)

        g_scld = { r : g[r] * s_k        for r in g }
        l      = { r : l[r] + g_scld[r]  for r in g } 
        #l      = { r : max(l[r], 0)      for r in g } 
        
        if k == max_iters -1:
            with open("l_final.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["g", "t", "var", "multiplier"])
                for (g,t,var), val in sorted(l.items()):
                    w.writerow([g, t, var, val])
            print('Saved final multipliers to l_final.csv')
            
        #print("Lagrange multiplier values:", l)
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
        #print(k)
    t_end = perf_counter()
    print(f"Total LR time = {t_end - t_all_start:.2f}s, "f"avg iter time = {np.mean(iter_times):.2f}s")

# one extra evaluation to show the dual with the final multipliers
    results_final = pool.map(solver_function, [(i, l, max_iters) for i in range(len(Time_windows))])
    Lag_final, _ = get_lagrange_and_g(results_final, Time_windows)

    Lag_set.append(Lag_final)   # now you have the post-update dual
    Level_vals.append(q)        # keep lengths aligned for plotting

    print("\n############### Summary of results #################")
    print("Level values", Level_vals)
    print("Lagrangian values", Lag_set)
    #print("lambda values: ", l)
    pool.close()
    pool.join()
    
    #Plot Results
    import numpy as np
    import matplotlib.pyplot as plt

    dual  = np.array(Lag_set)
    lbest = np.maximum.accumulate(dual)
    iters = np.arange(len(dual))

    plt.figure()

    # Solid blue for L_best
    plt.plot(iters, lbest, label='Best Lagrangian (running max)', color='C1', linestyle='--')

    # Orange dashed for Dual
    plt.plot(iters, dual,  label='Dual at iteration k', color='C0', linestyle='-')

    # (optional) Level line in green
    level = np.array(Level_vals[:len(dual)])
    plt.plot(iters, level, label='Level Value', color='C2')

    plt.xlabel('Iteration'); plt.ylabel('Value')
    plt.title('Dual and Best Lagrangian Lower Bound Over Iterations')
    plt.legend(); plt.tight_layout()
    plt.savefig('dual_vs_lbest_gamma={gamma}_hat={gamma_hat}_bounds=({y_LB},{y_UB})_T={T}.svg')                 # or: plt.savefig('dual_vs_lbest.svg', format='svg')
    plt.show()


    dual  = np.array(Lag_set)
    level = np.array(Level_vals)
    n     = min(len(dual), len(level))
    dual  = dual[:n]
    level = level[:n]
    iters = np.arange(n)
    plt.figure()
    plt.plot(iters, dual, label='Dual (Lagrange) Values')
    plt.plot(iters, level, label='Level Values')
    plt.xlabel('Iteration')
    plt.ylabel('Values')
    plt.title(f'Dual and Level Values Over Iterations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'dual_and_level_values_y={abs(y_UB)}.png')
    plt.show()

    g_l  = np.array(g_length)
    n     = len(g_length)
    levg_lel = level[:n]
    iters = np.arange(n)
    plt.figure()
    plt.plot(iters, g_l, label='Length of g')
    plt.xlabel('Iteration')
    plt.ylabel('Values')
    plt.title(f'Norm of g')
    plt.legend()
    plt.tight_layout()
    plt.savefig('g_length.png')
    plt.show()

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
