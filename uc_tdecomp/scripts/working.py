#working.py

from ..data.data_extract import load_json_data
import pandas as pd

dat = load_json_data(r"C:\Users\vdiazpa\Documents\quest_planning\quest_planning\seismic_model\datasets\RTS_GMLC_zonal_noreserves.json")
gen_data  = pd.read_csv(r"C:\Users\vdiazpa\Documents\quest_planning\quest_planning\seismic_model\datasets\RTS_data\gen_data.csv")

gens_in_csv = gen_data["GEN UID"]
gens_in_json = dat["gens"]
diff = set(gens_in_csv) - set(gens_in_json)

print("Json has this amt of gens:", len(gens_in_json))
print("gens in csv but not in json:", diff)

ren_gens_json   = []

generators = dat["elements"].get("generator", {})
for G in generators:
    gen = generators[G]
    if gen.get("generator_type") == "renewable":
        ren_gens_json.append(G)

print("there are", len(ren_gens_json), "renewable generators in json file") 

# Load time series data
rtpv_data  = pd.read_csv(r"C:\Users\vdiazpa\Documents\quest_planning\quest_planning\seismic_model\datasets\RTS_data\DAY_AHEAD_rtpv.csv")
pv_data    = pd.read_csv(r"C:\Users\vdiazpa\Documents\quest_planning\quest_planning\seismic_model\datasets\RTS_data\DAY_AHEAD_pv.csv")
hydro_data = pd.read_csv(r"C:\Users\vdiazpa\Documents\quest_planning\quest_planning\seismic_model\datasets\RTS_data\DAY_AHEAD_hydro.csv")
wind_data  = pd.read_csv(r"C:\Users\vdiazpa\Documents\quest_planning\quest_planning\seismic_model\datasets\RTS_data\DAY_AHEAD_wind.csv")

rtpv = rtpv_data.columns
pv = pv_data.columns
hydro = hydro_data.columns
wind = wind_data.columns

units = rtpv.append(pv)
units = units.append(hydro)
units = units.append(wind)
drops = ['Month', 'Day', 'Period', 'Year']
rens_ts = [u for u in units if u not in drops]
print("time series data accounts for this amt of renewable unis:", len(rens_ts))

ren_gens_csv = [gen_data["GEN UID"][i] for i in range(0,len(gen_data)) if gen_data["Fuel"][i] in ['Wind', 'Solar', 'Hydro']]
print("There are this amt of renewbale gens in csv:", len(ren_gens_csv))

print("Unit", set(ren_gens_json) - set(rens_ts), "is missing in time series data")