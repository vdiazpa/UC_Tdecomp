#run_bench.py

from ..data.data_extract import load_csv_data, load_rts_data
from ..opt.bench_UC import benchmark_UC_build

T = 72

#data =  load_csv_data(T)
#data = load_uc_data("./RTS_GMLC_zonal_noreserves.json")
data = load_rts_data(T)

x = benchmark_UC_build(data, opt_gap=0.001, tee = True, save_sol = False)
