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
# #file_path  = "examples/unit_commitment/RTS_GMLC_zonal_noreserves.json"
# #file_path  = "examples/unit_commitment/tiny_RTS_ready.json"
# #data = load_uc_data(file_path)
# data =  load_csv_data(T)

data = load_rts_data(T)

#print(dat)

# ###################################################################################

benchmark_UC_build(data, opt_gap = opt_gap, tee=True)

commitment, ofv, sol_to_plot = run_RH(data, F = F, L = L, T = T, write_csv = True, opt_gap = opt_gap, verbose = True, benchmark=False)


