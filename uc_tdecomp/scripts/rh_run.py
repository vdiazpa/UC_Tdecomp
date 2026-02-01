#rh_run.py

from ..data import load_csv_data, load_json_data, attach_battery_from_csv, attach_timeseries_from_rts_csv
from ..opt.RH_main import run_RH

dat = load_json_data(r"C:\Users\vdiazpa\Documents\quest_planning\quest_planning\seismic_model\datasets\RTS_GMLC_zonal_noreserves.json")
dat = attach_battery_from_csv(dat, r"C:\Users\vdiazpa\Documents\quest_planning\quest_planning\seismic_model\datasets\RTS_data\storage.csv")

# now override horizon + time series
dat = attach_timeseries_from_rts_csv(
    dat,
    load_csv=r"...\DAY_AHEAD_load.csv",
    ren_csvs={
        "pv":   r"C:\Users\vdiazpa\Documents\quest_planning\quest_planning\seismic_model\datasets\RTS_data\DAY_AHEAD_pv.csv",
        "rtpv": r"C:\Users\vdiazpa\Documents\quest_planning\quest_planning\seismic_model\datasets\RTS_data\DAY_AHEAD_rtpv.csv",
        "wind": r"C:\Users\vdiazpa\Documents\quest_planning\quest_planning\seismic_model\datasets\RTS_data\DAY_AHEAD_wind.csv",
        "hydro": r"C:\Users\vdiazpa\Documents\quest_planning\quest_planning\seismic_model\datasets\RTS_data\DAY_AHEAD_hydro.csv",
    },
    start_row=0,     # hour index in the 8760 file
    T=168,           # e.g. 1 week horizon
    strict=True)


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
