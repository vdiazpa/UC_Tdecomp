#rh_run.py

from ..data import load_csv_data
from ..opt.RH_main import run_RH


L = 8             # Lookahead
F = 8             # Roll forward period
T = 72             # length of planning horizon
prt_cry = False    # Print carryover constraints#
opt_gap = 0.01     # Optimality gap for monolithic solve
RH_opt_gap = 0.05  # Optimality gap for RH subproblems

################################### Load data #####################################
#file_path  = "examples/unit_commitment/RTS_GMLC_zonal_noreserves.json"
#file_path  = "examples/unit_commitment/tiny_RTS_ready.json"
#data = load_uc_data(file_path)
data =  load_csv_data(T)
###################################################################################


commitment, ofv, sol_to_plot = run_RH(data, F = F, L = L, T = T, write_csv = True, opt_gap = opt_gap, verbose = True, benchmark=False)
