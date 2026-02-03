#run_rh.py
from ..data.data_extract import load_csv_data, load_rts_data
from ..opt.RH_main import run_RH
from ..opt.bench_UC import benchmark_UC_build

RH_opt_gap = 0.05  # Optimality gap for RH subproblems
prt_cry = False    # Print carryover constraints#
opt_gap = 0.05     # Optimality gap for monolithic solve
L = 12             # Lookahead
F = 24            # Roll forward period
T = 72             # length of planning horizon

# ################################### Load data #####################################

#data = load_rts_data(T)
#print(data)
data = load_csv_data(T)
# ###################################################################################

def print_system_totals(data, T=None):
    periods = data["periods"]
    if T is not None:
        periods = periods[:T]

    demand = data.get("demand", {})
    ren_output = data.get("ren_output", {})
    ren_gens = data.get("ren_gens", [])

    for t in periods:
        total_demand = sum(v for (b, tt), v in demand.items() if tt == t)
        total_ren    = sum(ren_output.get((g, t), 0.0) for g in ren_gens)
        print(f"t={t:>3}  demand={total_demand:10.2f}  ren={total_ren:10.2f}")
#print_system_totals(data, T=5)


commitment, ofv, sol_to_plot = run_RH(
    data, F=F, L=L, T=T, 
    #s_tee = True,
    #write_csv = f"RH_sol_RTS_T{T}_F{F}_L{L}_gap{opt_gap}.csv", 
    RH_opt_gap=RH_opt_gap, verbose=True, benchmark=False)

benchmark_UC_build(data, opt_gap=opt_gap, tee=True)
