
# data_io.py
from uc_tdecomp.data_utils import _read_from_file
import pandas as pd

def load_uc_data(json_path: str):

    md = _read_from_file(json_path, file_type="json")
    system     = md.get("system", {})
    elements   = md.get("elements", {})
    periods    = list(system["time_keys"])             # e.g., [1,2,...,T] or actual labels
    periods    = [int(x) for x in periods]
    gens       = tuple(sorted(elements["generator"].keys()))
    T          = len(periods)
    loads      = elements.get("load", {})
    buses      = [key for key in elements.get("bus", {})]
    no_loads   = [f  for f  in buses if f not in loads]
    load_buses = [key for key in elements.get("load", {})]
    branches   = elements.get("branch")
    line_id    = [b for b in branches]

    k = 0
    lines = []
    line_capacity, line_reactance = {}, {}
    for branch in branches.values():
        line_capacity[line_id[k]]  = branch["rating_long_term"]
        line_reactance[line_id[k]] = branch["reactance"]
        lines.append((branch["from_bus"], branch["to_bus"]))
        k+=1
    line_endpoints = {line_id[i]: (lines[i][0], lines[i][1]) for i in range(len(lines))}

    #Line to bus 
    line_to_bus = pd.DataFrame(0, index = range(len(lines)), columns = list(buses))
    bus_to_unit = pd.DataFrame(0, index = list(gens), columns = list(buses))

    for i in range(len(lines)): 
        line_to_bus.loc[i, lines[i][0]] =-1
        line_to_bus.loc[i, lines[i][1]] = 1

    lTb_dict = {}
    for lid, (fb, tb) in line_endpoints.items():
        lTb_dict[(lid, fb)] = -1
        lTb_dict[(lid, tb)] =  1
        
    lines_by_bus = { b: tuple(l for l in line_id if (l, b) in lTb_dict) for b in buses }     #bus: tuple(lines_adjacent)

    # Get demand data
    demand = {}
    if loads: 
        for L in loads.values():
            bus  = L["bus"]
            vals = L["p_load"]["values"]  # assumed length T
            n = min(T, len(vals))
            for t in range(1,n+1):
                demand[(bus,t)] = float(vals[t-1])

    ref_bus, _ = max(demand, key = demand.get)

    ther_gens_dict = {}
    ren_gens   = []
    ther_gens  = []
    ren_output = {}
    generators = elements.get("generator", {})
    for G in generators:
        gen = generators[G]
        bus_to_unit.loc[G, gen.get("bus")] = 1
        if gen.get("generator_type") == "renewable":
            ren_gens.append(G)
            for t in range(T):
                val = gen['p_min']['values'][t]
                ren_output[(G, t+1)] = float(val) 
        else:
            ther_gens.append(G)
            ther_gens_dict[G] = gen

    #cost objects
    pw_slope, pw_intercept, commit_cost, start_cost = {}, {}, {}, {}

    for g, rec in ther_gens_dict.items():
        pc  = rec.get('p_cost', {})
        sc  = rec.get('startup_cost', {})
        vals = pc.get('values', [])
    
        commit_cost[g] = vals[0][1]
        start_cost[g]  =   sc[0][1]

        if not vals or len(vals) < 2: 
            continue

        B   = [float(MW) for MW, _ in vals]
        C   = [float(c)  for  _, c in vals] 

        slopes = []
        intercepts = []
        for l in range(1, len(B)):
            s_l = (C[l] - C[l-1]) / (B[l] - B[l-1])
            slopes.append(round(s_l,2))
            intercepts.append(round(C[l] - s_l * B[l], 2))

        pw_slope[g] = slopes
        pw_intercept[g] = intercepts

    # unit to bus mapping
    gen_bus     = {g: elements["generator"][g]["bus"] for g in gens}                 # dict with -> gen: bus
    gens_by_bus = {b: tuple(g for g in gens if gen_bus[g] == b) for b in buses}      # dict with -> bus: gens in bus

    ther_gens_by_bus = {b: tuple( g for g in ther_gens if gen_bus[g] == b) for b in buses}
    ren_bus_t = { (b,t): sum(ren_output[(g,t)] for g in ren_gens if gen_bus[g]==b) for b in buses for t in periods}

    # generator params
    p_max  = {g: elements["generator"][g]["p_max"] for g in ther_gens}
    p_min  = {g: elements["generator"][g]["p_min"] for g in ther_gens}
    s_init = {g: elements["generator"][g]["initial_status"] for g in ther_gens}
    rup    = {g: elements["generator"][g]["ramp_up_60min"] for g in ther_gens}
    rdn    = {g: elements["generator"][g]["ramp_down_60min"] for g in ther_gens}
    typ    = {g: elements["generator"][g]["fuel"] for g in gens}
    p_init = {g: elements["generator"][g]["initial_p_output"] for g in ther_gens}
    min_UT = {g: elements["generator"][g]["min_up_time"] for g in ther_gens}
    min_DT = {g: elements["generator"][g]["min_down_time"] for g in ther_gens}
    slp    = {g: pw_slope[g] for g in ther_gens if g in pw_slope}
    intc   = {g: pw_intercept[g] for g in ther_gens if g in pw_intercept}
    #suC    = {g: elements["generator"][g]["startup_cost"] for g in ther_gens}
    #sdC    = {g: elements["generator"][g]["shutdown_cost"] for g in ther_gens}

    # Ramping
    suR, sdR = {}, {}
    for g in ther_gens:
        gen_data = elements["generator"][g]
        suR[g] = gen_data.get("startup_capacity", gen_data.get("ramp_up_60min", 0))
        sdR[g] = gen_data.get("shutdown_capacity", gen_data.get("ramp_down_60min", 0))

    return {
        "system": system,
        "elements": elements,
        "periods": periods,
        "gens": gens,
        "buses": list(buses),
        "gen_bus": gen_bus,
        "gens_by_bus": gens_by_bus,
        "ref_bus": ref_bus, 
        "p_max": p_max,
        "p_min": p_min,
        "min_UT": min_UT,
        "min_DT": min_DT,
        "lines_by_bus": lines_by_bus,
        "ren_bus_t": ren_bus_t,
        "demand": demand,
        "lines" : line_id, 
        "line_ep": line_endpoints, 
        "ther_gens_by_bus": ther_gens_by_bus,
        "no_load_buses": no_loads,
        "line_cap": line_capacity, 
        "line_reac": line_reactance, 
        "init_status": s_init, 
        "p_init": p_init,
        "load_buses": load_buses,
        "lTb": lTb_dict, 
        "bTu": bus_to_unit, 
        "slp": slp,
        "intc": intc,
        # "n_segments": len(B)
        "startup_cost": start_cost,
        "commit_cost" : commit_cost,
        # "shutdown_cost": sdC,
        "rup": rup,
        "rdn": rdn,
        "suR": suR,
        "sdR": sdR,
        "type"  : typ,
        "ren_output": ren_output,
        "ren_gens": ren_gens,
        "ther_gens": ther_gens
    }


def load_csv_data(T): 
    
    data_folder_name = "./duke_data_interim_P1"
    
    line_thing = "/" if "/" in data_folder_name else "\\"
    gen_data    = pd.read_csv(f"{data_folder_name}{line_thing}data_genparams_partial_Interim_P1.csv", header=0)
    line_data   = pd.read_csv(f"{data_folder_name}{line_thing}line_params_new.csv", header=0)
    linetobus   = pd.read_csv(f"{data_folder_name}{line_thing}line_to_bus.csv", header=0)
    bus_to_unit = pd.read_csv(f"{data_folder_name}{line_thing}gen_mat_Interim_P1.csv", header=0)
    bus_to_unit.set_index("name", inplace=True)
    nuclear_df  = pd.read_csv(f"{data_folder_name}{line_thing}data_nuc_Interim_P1.csv", header=0)
    hydro_df    = pd.read_csv(f"{data_folder_name}{line_thing}data_hydro_H.csv", header=0)
    load_df     = pd.read_csv(f"{data_folder_name}{line_thing}data_load_2023.csv", header=0)
    solar_df    = pd.read_csv(f"{data_folder_name}{line_thing}data_solar_2023.csv", header=0)
    wind_df     = pd.read_csv(f"{data_folder_name}{line_thing}data_wind_2023.csv", header=0)
    sto_mat     = pd.read_csv(f"{data_folder_name}{line_thing}storage_mat_Interim_P1.csv", header=0)
    sto_params  = pd.read_csv(f"{data_folder_name}{line_thing}data_batparams_Interim_P1.csv", header=0)


    periods      = [i for i in range(1,T+1,1)]
    gens         = gen_data["name"].tolist()
    all_nodes    = list(linetobus.columns[1:])
    nodes_load   = list(load_df.columns)
    nodes_noload = [n for n in all_nodes if n not in nodes_load]
    lines_list   = linetobus["line"].to_list()
    lines        = [("n_" + l.split("_")[1], "n_" + l.split("_")[3]) for l in lines_list]
    
    line_reactance = dict(zip(line_data["line"], line_data["reactance"]))
    line_capacity  = {line_data["line"][i]: line_data["limit"][i] for i in range(len(lines))}
    line_endpoints = {lines_list[i]: (lines[i][0], lines[i][1]) for i in range(len(lines))}

    line_to_bus  = linetobus.iloc[:, 1:]
    line_to_bus.rename(index={i: lines_list[i] for i in range(len(lines_list))}, inplace=True)
    line_to_bus_dict = {(lid, b): line_to_bus.loc[lid, b] for lid in lines_list for b in all_nodes if line_to_bus.loc[lid, b] != 0}
    bus_to_unit_dict = {(lid, b): bus_to_unit.loc[lid, b] for lid in gens       for b in all_nodes if bus_to_unit.loc[lid, b] != 0}

    lines_by_bus = { b: tuple(l for l in lines_list if (l, b) in line_to_bus_dict) for b in all_nodes }

    #gen -> bus mapping from gen_mat
    gen_bus = {}
    for g in gens:
        if g in bus_to_unit.index:
            for b in all_nodes:
                if bus_to_unit.at[g,b] !=0:
                    gen_bus[g] = b
                    break

    #Get renewable output & demand time series
    ren_out_yr = pd.concat([solar_df, hydro_df, wind_df, nuclear_df], axis=1)

    ren_out_T = ren_out_yr.iloc[:T].copy()
    ren_out_T.index = periods

    ren_gens   = ren_out_T.columns.tolist()
    ther_gens  = [gen for gen in gens if gen not in ren_gens]
    ren_output = {(g, t): float(ren_out_T.loc[t, g]) for g in ren_gens for t in periods}
    load_T = load_df.iloc[:T].copy()
    load_T.index = periods
    demand = {(bus,t): float(load_T.loc[t,bus]) if bus in load_T.columns else 0.0 for bus in all_nodes for t in periods}

    ref_bus, _ = max(demand, key = demand.get)

    gens_by_bus, ther_gens_by_bus = {}, {}

    for bus in bus_to_unit.columns:
        gbb   = []
        gbb_t = []
        for g in bus_to_unit.index:
            if bus_to_unit.loc[g,bus] == 1:
                gbb.append(g)
                if g in ther_gens:
                    gbb_t.append(g)
        if len(gbb) !=0:
            gens_by_bus[bus] = tuple(gbb)
        if len(gbb_t) !=0:
            ther_gens_by_bus[bus] = tuple(gbb_t)

    ren_bus_t  = {(b,t): sum(ren_output[(g,t)] for g in ren_gens if gen_bus.get(g) == b) for b in all_nodes for t in periods}

    gen_th = gen_data[gen_data["name"].isin(ther_gens)].set_index("name") # filter gen data df for thermal gens
    p_max  = gen_th["maxcap"].to_dict()
    p_min  = gen_th["mincap"].to_dict()
    s_init = {g: 0   for g in ther_gens}
    p_init = {g: 0.0 for g in ther_gens} #assumes power generated at 0 is 0
    suR    = gen_th["ramp"].to_dict()
    sdR    = gen_th["ramp"].to_dict()
    rup    = gen_th["ramp"].to_dict()
    rdn    = gen_th["ramp"].to_dict()
    typ    = gen_th["typ"].to_dict()
    min_UT = gen_th["minup"].to_dict()
    min_DT = gen_th["mindn"].to_dict()

    start_cost  = gen_th["st_cost"].to_dict()
    commit_cost = gen_th["no_load"].to_dict()

    # slp    = {g: pw_slope[g] for g in ther_gens if g in pw_slope}
    # intc   = {g: pw_intercept[g] for g in ther_gens if g in pw_intercept}
    # suC    = {g: elements["generator"][g]["startup_cost"] for g in ther_gens}
    # sdC    = {g: elements["generator"][g]["shutdown_cost"] for g in ther_gens}

    # Dictionary to map nodes to lines adjacent & count number of lines adjacent to node
    lines_adj = {}
    bus_nlines_adj= {}
    for col in line_to_bus.columns:
        lines_adj[col] = []
        bus_nlines_adj[col] = 0
        for idx, line in enumerate(line_to_bus.index):
            if line_to_bus.loc[line, col] != 0:
                bus_nlines_adj[col] +=1
                lines_adj[col].append(lines_list[idx])

    # Dictionary to map units to buses
    unit_to_bus_dict = {}  
    gen_buses = []
    for idx, unit in enumerate(bus_to_unit.index):  
        for col in bus_to_unit.columns:  
            if bus_to_unit.loc[unit, col] != 0:
                unit_to_bus_dict[unit] = col  
                if col not in gen_buses:
                    gen_buses.append(col)
                    
    #Storage Parameters
    bats = sto_params["name"].tolist()
    sto_params["node_bat"].apply(lambda x: f"n_{str(x)}")
    sto_params.set_index("name", inplace=True)
    sto_RoC  = sto_params["bat_RoC"].to_dict()
    sto_Ecap = sto_params["bat_cap"].to_dict()
    sto_eff  = sto_params["bat_eff"].to_dict()
    SoC_init = {b: 0.0 for b in bats}  #assumes storage initial SoC is 0
    
    bat_bus = sto_params["node_bat"].apply(lambda x: f"n_{str(x)}").to_dict()
    
    df = sto_mat.set_index('name')
    
    bus_bat = {}
    for bus in all_nodes:
        if df[bus].sum() > 0:
            l = []
            for b in bats:
                if df[bus][b] >0:
                    l.append(b)
            bus_bat[bus] = l
                
    return {
        "gens": gens,
        "bats": bats,
        "SoC_init": SoC_init,
        "bus_bat": bus_bat,
        "bat_bus": bat_bus,
        "sto_RoC": sto_RoC,
        "sto_Ecap": sto_Ecap,
        "sto_eff": sto_eff,
        "buses": list(all_nodes),
        "periods": periods,
        "p_max": p_max,
        "p_min": p_min,
        "min_UT": min_UT,
        "min_DT": min_DT,
        "demand": demand,
        "lines" : lines_list, 
        "ref_bus" : ref_bus,
        "line_ep": line_endpoints, 
        "no_load_buses": nodes_noload,
        "line_cap": line_capacity, 
        "line_reac": line_reactance, 
        "lines_by_bus": lines_by_bus, 
        "gens_by_bus": gens_by_bus, 
        "gen_bus": gen_bus,
        "ther_gens_by_bus": ther_gens_by_bus,
        "ren_bus_t": ren_bus_t,
        "init_status": s_init, 
        "p_init": p_init,
        "load_buses": nodes_load,
        "lTb": line_to_bus_dict, 
        "bTu": bus_to_unit, 
        # "slp": slp,
        # "intc": intc,
        "startup_cost": start_cost,
        "commit_cost" : commit_cost,
        # "n_segments": len(B)
        # "shutdown_cost": sdC,
        "rup": rup,
        "rdn": rdn,
        "suR": suR,
        "sdR": sdR,
        "type"  : typ,
        "ren_output": ren_output,
        "ren_gens": ren_gens,
        "ther_gens": ther_gens
    }
