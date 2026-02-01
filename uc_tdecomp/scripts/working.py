#working.py

from ..data.data_extract import load_json_data
import pandas as pd

dat = load_json_data(r"C:\Users\vdiazpa\Documents\quest_planning\quest_planning\seismic_model\datasets\RTS_GMLC_zonal_noreserves.json")
gen_data  = pd.read_csv(r"C:\Users\vdiazpa\Documents\quest_planning\quest_planning\seismic_model\datasets\RTS_data\gen_data.csv")

gens_in_csv = gen_data["GEN UID"]
gens_in_json = dat["gens"]

print("Json has this amt of gens:", len(gens_in_json))

diff = set(gens_in_csv) - set(gens_in_json)

print("gens in csv but not in json:", diff)