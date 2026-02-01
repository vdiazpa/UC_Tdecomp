#bench_run.py

from ..data.data_extract import load_csv_data
from ..opt.bench_UC import benchmark_UC_build

T = 24
file_path  = "./RTS_GMLC_zonal_noreserves.json"
#file_path = "examples/unit_commitment/tiny_rts_ready.json"

data =  load_csv_data(T)
#data = load_uc_data(file_path)

x = benchmark_UC_build(data, opt_gap=0.01, tee = True, save_sol = False)
