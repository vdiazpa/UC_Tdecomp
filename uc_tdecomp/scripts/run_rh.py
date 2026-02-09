#run_rh.py
from ..data.data_extract import load_csv_data, load_rts_data
from ..opt.RH_main import run_RH
from ..opt.bench_UC import benchmark_UC_build, plot_soc_overlay_top_active_3_tight

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
        
# ################################### Parameters ####################################
T = 72
RH_opt_gap = 0.01  # Optimality gap for RH subproblems
prt_cry = False    # Print carryover constraints#
opt_gap = 0.01     # Optimality gap for monolithic solve
L = 12             # Lookahead
F = 24            # Roll forward period
MUT = 'counter'

data = load_csv_data(T)

rh_time, ofv, commitment = run_RH(
data, F=F, L=L, T=T, 
#s_tee = True,
#write_csv = f"RH_sol_RTS_T{T}_F{F}_L{L}_gap{opt_gap}_MT{MUT}.csv", 
RH_opt_gap=RH_opt_gap, verbose=True, benchmark=False, MUT = "counter", MDT = "counter", plt_soc=False)
    
return_full = benchmark_UC_build(data, opt_gap=0.01, MUT='counter', MDT='counter', tee=False)
return_rh = benchmark_UC_build(data, opt_gap=0.01, fixed_commitment=commitment, MUT='counter', MDT='counter', tee=False)

top_units = plot_soc_overlay_top_active_3_tight(
    return_full, return_rh,
    out_path=f"SoC_Comparison/SoC_overlay_top3_T{T}_F{F}_L{L}.pdf",
    soc_units="SoC (MWh)",      # change if your units differ
    figsize=(3.5, 2.35)         # tweak taller if y-ticks still feel tight
)

print("Plotted top active units:", top_units)

# for T in [72, 168, 336]:
#     data = load_rts_data(T)
#     commitment, ofv, sol_to_plot = run_RH(
#     data, F=F, L=L, T=T, 
#     #s_tee = True,
#     #write_csv = f"RH_sol_RTS_T{T}_F{F}_L{L}_gap{opt_gap}_MT{MUT}.csv", 
#     RH_opt_gap=RH_opt_gap, verbose=True, benchmark=False, MUT = "counter", MDT = "counter")

    #benchmark_UC_build(data, opt_gap=opt_gap, tee=False, MUT='classical', MDT='classical')


# for T in [72, 168, 336]:
#     data = load_csv_data(T)
#     commitment, ofv, sol_to_plot = run_RH(
#     data, F=F, L=L, T=T, 
#     #s_tee = True,
#     #write_csv = f"RH_sol_DUK_T{T}_F{F}_L{L}_gap{opt_gap}_MT{MUT}.csv", 
#     RH_opt_gap=RH_opt_gap, verbose=True, benchmark=False, MUT = "counter", MDT = "counter")

    #benchmark_UC_build(data, opt_gap=opt_gap, tee=False, MUT='classical', MDT='classical')

