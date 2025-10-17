
from uc_tdecomp.LR_subp_build_t import build_subprobs_t
from uc_tdecomp.data_extract import load_uc_data, load_csv_data
from uc_tdecomp.bench_UC import benchmark_UC_build
import numpy as np
import math
from pyomo.environ import *
from multiprocessing import Pool

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

def make_tasks_for_pool(T_windows, lambda_obj, k):
    return [(i,lambda_obj, k) for i in range(len(T_windows))]

models  = [] # list of models, one per time window
solvers = [] # list of persistent solvers

def initialize_model(data, Time_windows, index_set):
    global models, solvers
    models  = []
    solvers = []
    for s_e in Time_windows:
        m = build_subprobs_t(data, s_e, index_set=index_set)
        opt = SolverFactory('gurobi_persistent')
        opt.options['LogToConsole'] = 0
        opt.options['OutputFlag'] = 0
        opt.set_instance(m)
        opt.options['Presolve']  = 2  
        opt.options['Seed']      = seed
        opt.options['MIPGap']    = 0.3
        solvers.append(opt)
        models.append(m)

def solver_function(task):
    sub_id, lamda_obj, k = task
    m   = models[sub_id]
    opt = solvers[sub_id]

    # Update lagrange multipliers in the model
    for key, val in lamda_obj.items():
        if key in m.L_index:
            m.L[key] = val

    result = solve_and_return(m, opt, k)
    return result

seed = 345

if __name__ == "__main__":

    #file_path  = "examples/unit_commitment/RTS_GMLC_zonal_noreserves.json"
    #file_path  = "examples/unit_commitment/tiny_rts_ready.json"
    subn = 24  # number of periods in each subproblem
    seed = 345

    #data        = load_uc_data(file_path)
    data        =  load_csv_data(48)
    ther_gens   = data["ther_gens"]
    num_periods = len(data["periods"])
    num_sh      = math.ceil(num_periods/subn)       # number of subproblems

    Time_windows = [list(range(i, min(i+subn, num_periods+1))) for i in range(1, num_periods+1, subn)]


    print("num_periods:", num_periods)
    print("subn:", subn)
    print("Time_windows:", Time_windows)
    print("#subproblems:", len(Time_windows))

    index_set = set()
    for i in range(subn, num_periods, subn):
        for var in ['UnitOn', 'PowerGenerated', 'DT_Obl', 'UT_Obl']:
            for g in ther_gens: 
                    index_set.add((g, i, var))

    # Initialize 
    y_LB, y_UB = -np.inf, np.inf
    pow_pnlty, commit_pnlty, UT_pnlty, DT_pnlty = 35, 30, 40, 32
    max_iters  = 15

    l = {}
    for key in index_set:
        if 'UnitOn' in key:
            l[key] = commit_pnlty
        elif 'PowerGenerated' in key:
            l[key] = pow_pnlty
        elif 'DT_Obl' in key:
            l[key] = DT_pnlty
        else:
            l[key] = UT_pnlty

    q0 = benchmark_UC_build(data, 0.95, seed = seed)['ofv'] 

    results0 = []
    for s_e in Time_windows:
        m = build_subprobs_t(data, s_e, index_set=index_set)
        opt = SolverFactory('gurobi_persistent')
        opt.set_instance(m)
        opt.options['Presolve'] = 2  
        opt.options['MIPGap']   = 0.3
        opt.options['Seed']     = seed
        results0.append(solve_and_return(m, opt, 0))

    Lag0, g0  =  get_lagrange_and_g(results0, Time_windows)

    print("qbar is: ", f"{q0:.2f}")
    print("Dual Value is", f"{Lag0:.2f}")

    n_workers = len(Time_windows)
    pool = Pool(processes=n_workers, initializer = initialize_model, initargs = (data, Time_windows, index_set))

    # Initialize parameters for the main loop
    Lag_set, g_set, Level_vals = [], [], []       # tracks g, Lagrange, level values
    # gamma      = 1/num_sh   # this is 1/8 = 0.125
    gamma      = 0.5
    gamma_hat  = gamma + 1  # this is 1.125  
    Lag, q, g  = 0, 0, 0  
    k_j        = [0]           # tracks iteration of level updates 
    eta        = 0             # number of iterations before last level update
    j          = 0             # number of level updates
    g_length   = []
    g_history  = []

    # # Initialize PSVD model
    lam_len             = range(len(g0))
    PSVD                = ConcreteModel() 
    PSVD.y              = Var( index_set, within = Reals, bounds = (y_LB, y_UB))
    PSVD.inf_detect_cut = ConstraintList(doc = 'inf_detect')

    opt = SolverFactory('gurobi')

    # Step 1. Main Loop
    k = 0
    while k < max_iters:
        if k == 0:
            q    = q0 
            g    = g0
            Lag  = Lag0

        else: 
            # results = pool.map(solve_subproblem, update_args_list(l, args_list, k)) # solves and stores
            tasks = make_tasks_for_pool(Time_windows, l, k)
            results = pool.map(solver_function, tasks)            # solves and stores
            Lag, g  = get_lagrange_and_g(results, Time_windows)
            
        g_norm = np.linalg.norm(np.array(list(g.values())), ord = 2)
        Lag_set.append(Lag)
        g_set.append(g)
        diff   = q - Lag 
        s_k    = gamma * (diff / g_norm**2 ) #stepsize

        g_history.append(g)
        g_length.append(g_norm)
        print("\ng length is:", f"{g_norm:.4f}", "\nStepsize is:", f"{s_k:.2f}")# , "\nG is:", g)

        PSVD.inf_detect_cut.add( sum( g[r] * PSVD.y[r] for r in index_set ) <= sum( g[r] * l[r] for r in index_set ) - (1/gamma_hat) * s_k * g_norm**2 )
        #PSVD.inf_detect_cut.pprint()

        PSVD_results = opt.solve(PSVD, tee=False)

        g_scld = { r : g[r] * s_k        for r in g }
        l      = { r : l[r] + g_scld[r]  for r in g } 
        #l      = { r : max(l[r], 0)      for r in g } 

        #print("Lagrange multiplier values:", l)
        print("PSVD results:", PSVD_results.solver.termination_condition )

        print("Dual value at iteration", k, "is:", f"{Lag:.2f}")
        
        #print(f"\nTotal RH time: {rh_time:.3f} secs")

        if PSVD_results.solver.termination_condition != TerminationCondition.optimal:
            print("Infeasible PSVD solution detected. Adjusting step size.")
            q   = (gamma/gamma_hat) * q + (1 - gamma/gamma_hat) * max(Lag_set[ k_j[-1] : k_j[-1]+eta+1 ])
            print("New q value is:", f"{q:.2f}")
            PSVD.inf_detect_cut.clear()  # clear the constraint list 
            eta = 0
            Level_vals.append(q)
            k_j.append(k)
            j+=1

        else:
            eta += 1
            print(f"q value in iteration {k} is:", f"{q:.2f}")
            Level_vals.append(q)

        k+=1
        #print(k)

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
