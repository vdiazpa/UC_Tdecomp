#rh_run.py

from ..data.data_extract import load_csv_data, load_rts_data
from ..opt.RH_main import run_RH
from ..opt.bench_UC import benchmark_UC_build

L = 12             # Lookahead
F = 72             # Roll forward period
T = 168             # length of planning horizon
prt_cry = False    # Print carryover constraints#
opt_gap = 0.01     # Optimality gap for monolithic solve
RH_opt_gap = 0.05  # Optimality gap for RH subproblems

# ################################### Load data #####################################

data = load_rts_data(T)
#print(data)

# ###################################################################################

#benchmark_UC_build(data, opt_gap = opt_gap, tee=True)

commitment, ofv, sol_to_plot = run_RH(data, F = F, L = L, T = T, 
            write_csv = f"RH_sol_RTS_T{T}_F{F}_L{L}_gap{opt_gap}.csv", opt_gap = opt_gap, verbose = True, benchmark=False)


