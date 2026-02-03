#run_benders.py
from ..data.data_extract import  load_csv_data, load_rts_data
from ..opt.benders import model_build_solve_benders_mpi, solve_MILP
from ..opt.bench_UC import benchmark_UC_build
from time import perf_counter

# ============================ Main

T = 24

data =  load_csv_data(T)
#data = load_uc_data(file_path)

#M, SP = run_benders(data, max_iter=10, tol=1e-2)

# print("\n################### Done with manual Benders. Solve using mpi-sppy #########################")
#model_build_solve_benders_mpi(data, max_iter = 10, fixed_commitment = None)
# print("\n########################## Done with Benders. Solve as MILP ################################")
# solve_MILP(benchmark_UC_build(data), tee=True, opt_gap = 0.001)

# print("\n################### Solv using mpi-sppy, all ints on first stage #########################")
# t1 = perf_counter()
# model_build_solve_benders_mpi(data, max_iter = 10, fixed_commitment = None)
# run1_time = perf_counter() - t1
# print("\n################### Finished solve using mpi-sppy w9ith all ints on first stage #########################")
# print(f"Total time with all ints on first stage: {run1_time:.3f} secs")

# print("\n################### Solve using mpi-sppy, commit only on first stage #########################")
# def scenario_creator(scenario_name, **kwargs):
    
#     data = kwargs.get("data")
#     fixed_commitment = kwargs.get("fixed_commitment")
#     m = benchmark_UC_build(data, fixed_commitment=fixed_commitment)
#     sputils.attach_root_node(m, m.StageOneCost, [m.UnitOn]) # ---- Attach Non-anticipativity info ---
#     m._mpisppy_probability = 1.0 

#     return m  
      
t2 = perf_counter()
model_build_solve_benders_mpi(data,  fixed_commitment = None)
run2_time = perf_counter() - t2
print("\n################### Finished solve using mpi-sppy w9ith commit only on first stage #########################")
print(f"Total time with commit only on first stage: {run2_time:.3f} secs")

print("\n########################## Done with Benders. Solving as MILP (B&C) ################################")
solve_MILP(benchmark_UC_build(data), tee=False, opt_gap = 0.01, do_solve=True)