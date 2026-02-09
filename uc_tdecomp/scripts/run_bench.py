#run_bench.py

from ..data.data_extract import load_csv_data, load_rts_data
from ..opt.bench_UC import benchmark_UC_build

T = 24             # length of planning horizon

#data =  load_csv_data(T)
#data = load_uc_data("./RTS_GMLC_zonal_noreserves.json")
data = load_rts_data(T)

benchmark_UC_build(data, opt_gap=0.01, tee = True, save_sol = "sol_from_bench_24hr_rts.csv", MUT = "counter", MDT = "counter")
