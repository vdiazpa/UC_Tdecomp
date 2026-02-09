#run_benders.py
from ..data.data_extract import  load_csv_data, load_rts_data
from ..opt.benders import model_build_solve_benders_mpi, solve_MILP
from ..opt.bench_UC import benchmark_UC_build
from time import perf_counter


# T = 72
# data =  load_csv_data(T)
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

for T in [72, 168, 336]: 
#T = 72
    data =  load_rts_data(T)
    m = benchmark_UC_build(data, tee=False, opt_gap = 0.01, do_solve=False)

    # print("\n########################## Solving as MILP (B&C) ################################")
    # t0 = perf_counter()
    # solve_MILP(m, opt_gap=0.01, tee=False)
    # run1_time = perf_counter() - t0
    # print(f"Total time with commit only on first stage: {run1_time:.3f} secs")


    print("\n######################### Solving with Benders ##################################")
    t2 = perf_counter()
    model_build_solve_benders_mpi(data,  fixed_commitment = None)
    run2_time = perf_counter() - t2
    print(f"Total time with commit only on first stage: {run2_time:.3f} secs")

