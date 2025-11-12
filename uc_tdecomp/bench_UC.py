from uc_tdecomp.data_extract import load_uc_data, load_csv_data
import numpy as np
from pyomo.environ import *
from time import perf_counter


T = 72
file_path  = "./RTS_GMLC_zonal_noreserves.json"
#file_path = "examples/unit_commitment/tiny_rts_ready.json"

data =  load_csv_data(T)
#data = load_uc_data(file_path)

def benchmark_UC_build(data, opt_gap, fixed_commitment=None, tee = False, save_sol = False, F = False, L = False):
    
    m   = ConcreteModel()
    t0  = perf_counter()
    opt = SolverFactory('gurobi')
    opt.warm_start_capable()

    # ======================== Sets

    m.TimePeriods         = data['periods'] 
    m.LoadBuses           = Set(initialize=data['load_buses'], ordered=True)
    m.InitialTime         = min(m.TimePeriods)
    m.FinalTime           = max(m.TimePeriods)
    m.ThermalGenerators   = Set(initialize=data['ther_gens'], ordered=True)
    m.RenewableGenerators = Set(initialize=data['ren_gens'],  ordered=True)
    m.Generators          = Set(initialize=data['gens'],      ordered=True)
    m.TransmissionLines   = Set(initialize=data['lines'],     ordered=True)
    m.StorageUnits        = Set(initialize = data['bats'], ordered=True)
    #m.CostSegments        = Set(initialize=range(1, data['n_segments']), ordered=True)  # number of piecewise cost segments

    # ======================== Parameters 

    W_full                  = m.FinalTime - m.InitialTime + 1
    m.MinUpTime          = Param(m.ThermalGenerators, initialize = data['min_UT'])
    m.MinDownTime        = Param(m.ThermalGenerators, initialize = data['min_DT'])
    m.PowerGeneratedT0   = Param(m.ThermalGenerators, initialize = data['p_init'])   
    m.StatusAtT0         = Param(m.ThermalGenerators, initialize = data['init_status'] ) 
    m.UnitOnT0           = Param(m.ThermalGenerators, initialize = lambda m, g: 1.0 if m.StatusAtT0[g] > 0 else 0.0)
    m.SoCAtT0            = Param(m.StorageUnits,      initialize = data['SoC_init']) 

    m.InitialTimePeriodsOnline = Param(m.ThermalGenerators, 
                                        initialize = lambda m, g: (min(W_full, max(0, int(m.MinUpTime[g]) - int(m.StatusAtT0[g]))) if m.StatusAtT0[g] > 0 else 0))
    
    m.InitialTimePeriodsOffline =  Param(m.ThermalGenerators, 
                                        initialize = lambda m, g: (min(W_full, max(0, int(m.MinDownTime[g]) - abs(int(m.StatusAtT0[g])))) if m.StatusAtT0[g] < 0 else 0))
    
    m.MaximumPowerOutput    = Param(m.ThermalGenerators,   initialize=data['p_max'])
    m.MinimumPowerOutput    = Param(m.ThermalGenerators,   initialize=data['p_min'])
    m.RenewableOutput       = Param(m.RenewableGenerators, data["periods"], initialize=data['ren_output'])
    m.CommitmentCost        = Param(m.ThermalGenerators,   initialize=data['commit_cost'])
    m.StartUpCost           = Param(m.ThermalGenerators,   initialize=data['startup_cost'])
    #m.ShutDownCost           = Param(m.ThermalGenerators,   initialize=data['shutdown_cost'])
    m.NominalRampUpLimit    = Param(m.ThermalGenerators,   initialize=data['rup'])
    m.NominalRampDownLimit  = Param(m.ThermalGenerators,   initialize=data['rdn'])
    m.StartupRampLimit      = Param(m.ThermalGenerators,   initialize=data['suR'])
    m.ShutdownRampLimit     = Param(m.ThermalGenerators,   initialize=data['sdR'])
    m.FlowCapacity          = Param(m.TransmissionLines,   initialize = data['line_cap'])
    m.LineReactance         = Param(m.TransmissionLines,   initialize = data['line_reac'])
    m.Storage_RoC           = Param(m.StorageUnits,        initialize = data['sto_RoC'])
    m.Storage_EnergyCap     = Param(m.StorageUnits,        initialize = data['sto_Ecap'])
    m.Storage_Efficiency    = Param(m.StorageUnits,        initialize = data['sto_eff'])

     # ======================== Variables 

    m.PowerGenerated      = Var(m.ThermalGenerators, m.TimePeriods, within=NonNegativeReals) 
    m.RenPowerGenerated   = Var(m.RenewableGenerators, m.TimePeriods, within=NonNegativeReals) 
    m.PowerCostVar        = Var(m.ThermalGenerators, m.TimePeriods, within=NonNegativeReals) 
    m.UTRemain            = Var(m.ThermalGenerators, m.TimePeriods, within=NonNegativeReals) 
    m.DTRemain            = Var(m.ThermalGenerators, m.TimePeriods, within=NonNegativeReals) 
    m.UnitOn              = Var(m.ThermalGenerators, m.TimePeriods, within=Binary)
    m.UnitStart           = Var(m.ThermalGenerators, m.TimePeriods, within=Binary)
    m.UnitStop            = Var(m.ThermalGenerators, m.TimePeriods, within=Binary)
    m.V_Angle             = Var(data['buses'],       m.TimePeriods, within = Reals, bounds = (-180, 180) )
    m.Flow                = Var(m.TransmissionLines, m.TimePeriods, within = Reals, bounds = lambda m, l, t: (-value(m.FlowCapacity[l]), value(m.FlowCapacity[l])))
    m.LoadShed            = Var(data["load_buses"],  m.TimePeriods, within = NonNegativeReals)
    m.SoC                 = Var(m.StorageUnits,      m.TimePeriods, within = NonNegativeReals, bounds = lambda m, b, t: (0, m.Storage_EnergyCap[b]) )
    m.ChargePower         = Var(m.StorageUnits,      m.TimePeriods, within = NonNegativeReals, bounds = lambda m, b, t: (0, m.Storage_RoC[b]) )
    m.DischargePower      = Var(m.StorageUnits,      m.TimePeriods, within = NonNegativeReals, bounds = lambda m, b, t: (0, m.Storage_RoC[b]) )
    m.IsCharging          = Var(m.StorageUnits,      m.TimePeriods, within = Binary)
    m.IsDischarging       = Var(m.StorageUnits,      m.TimePeriods, within = Binary)

    # ======================================= Single-period constraints ======================================= #

    m.MaxCapacity_thermal = Constraint(m.ThermalGenerators,   m.TimePeriods, rule=lambda m,g,t: m.PowerGenerated[g,t] <= m.MaximumPowerOutput[g]*m.UnitOn[g,t], doc= 'max_capacity_thermal')
    m.MinCapacity_thermal = Constraint(m.ThermalGenerators,   m.TimePeriods, rule=lambda m,g,t: m.PowerGenerated[g,t] >= m.MinimumPowerOutput[g]*m.UnitOn[g,t], doc= 'min_capacity_thermal')
    m.MaxCapacity_renew   = Constraint(m.RenewableGenerators, m.TimePeriods, rule=lambda m,g,t: m.RenPowerGenerated[g,t] <= m.RenewableOutput[(g,t)], doc= 'renewable_output')
    m.Power_Flow          = Constraint(m.TransmissionLines,   m.TimePeriods, rule=lambda m,l,t: m.Flow[l,t]*m.LineReactance[l] == m.V_Angle[data["line_ep"][l][0],t] - m.V_Angle[data["line_ep"][l][1],t], doc='Power_flow')

    def nb_rule(m, b, t):
        thermal = sum(m.PowerGenerated[g,t] for g in data["ther_gens_by_bus"].get(b, []))
        flows   = sum(m.Flow[l,t] * data['lTb'][(l,b)] for l in data['lines_by_bus'][b])
        renew   = 0.0 if b not in data['bus_ren_dict'] else sum(m.RenPowerGenerated[g,t] for g in data['bus_ren_dict'][b])
        shed    = m.LoadShed[b,t] if b in data["load_buses"] else 0.0
        storage = 0.0 if b not in data['bus_bat'] else sum(m.DischargePower[bat,t] - m.ChargePower[bat,t] for bat in data['bus_bat'][b])
        return thermal + flows + renew + shed + storage == data["demand"].get((b,t), 0.0)
    
    m.NodalBalance = Constraint(data["buses"], m.TimePeriods, rule = nb_rule)

    for t in m.TimePeriods:
        m.V_Angle[data["ref_bus"], t].fix(0.0)
        
    # ======================================= Logical, Ramping and MUT/MDT (counter) ======================================= #
    
    m.logical_constraints  = ConstraintList(doc = 'logical')
    m.RampUp_constraints   = ConstraintList(doc = 'ramp_up')
    m.RampDown_constraints = ConstraintList(doc = 'ramp_down')
    m.UTRemain_constraints = ConstraintList(doc = 'UT_remain')
    m.DTRemain_constraints = ConstraintList(doc = 'DT_remain')

    for g in m.ThermalGenerators:
        for t in m.TimePeriods: 
            m.UTRemain_constraints.add( m.UTRemain[g,t] <= m.MinUpTime[g] * m.UnitOn[g,t]) 
            m.DTRemain_constraints.add( m.DTRemain[g,t] <= m.MinDownTime[g] * (1 - m.UnitOn[g,t]))
            
            
            if t == m.InitialTime:
                m.logical_constraints.add(m.UnitStart[g,t] - m.UnitStop[g,t] == m.UnitOn[g,t] - m.UnitOnT0[g])
                m.RampUp_constraints.add(m.PowerGenerated[g,t] - m.PowerGeneratedT0[g]<= m.NominalRampUpLimit[g] * m.UnitOnT0[g] + m.StartupRampLimit[g] * m.UnitStart[g,t])
                m.RampDown_constraints.add(m.PowerGeneratedT0[g] - m.PowerGenerated[g,t] <= m.NominalRampDownLimit[g] * m.UnitOn[g,t] + m.ShutdownRampLimit[g] * m.UnitStop[g,t]) #assumes power generated at 0 is 0
                
                m.UTRemain_constraints.add( m.UTRemain[g,t] >=  m.InitialTimePeriodsOnline[g] +m.MinUpTime[g]*m.UnitStart[g,t] - m.UnitOn[g,t])
                m.DTRemain_constraints.add( m.DTRemain[g,t] >=  m.InitialTimePeriodsOffline[g] +m.MinDownTime[g]*m.UnitStop[g,t] - (1 -m.UnitOn[g,t]))

            else: 
                m.logical_constraints.add(m.UnitStart[g,t] - m.UnitStop[g,t] == m.UnitOn[g,t] - m.UnitOn[g,t-1])
                m.RampUp_constraints.add(m.PowerGenerated[g,t] - m.PowerGenerated[g,t-1] <= m.NominalRampUpLimit[g] * m.UnitOn[g, t-1] + m.StartupRampLimit[g] * m.UnitStart[g,t])
                m.RampDown_constraints.add(m.PowerGenerated[g,t-1] - m.PowerGenerated[g,t] <= m.NominalRampDownLimit[g] * m.UnitOn[g, t] + m.ShutdownRampLimit[g] * m.UnitStop[g,t])
                
                m.UTRemain_constraints.add( m.UTRemain[g,t] >=  m.UTRemain[g,t-1] +m.MinUpTime[g]*m.UnitStart[g,t] - m.UnitOn[g,t])
                m.DTRemain_constraints.add( m.DTRemain[g,t] >=  m.DTRemain[g,t-1] +m.MinDownTime[g]*m.UnitStop[g,t] - (1 -m.UnitOn[g,t]))

    for g in m.ThermalGenerators:
        
        lg = min(m.FinalTime, int(m.InitialTimePeriodsOnline[g]))         # CarryOver Uptime    
        if m.InitialTimePeriodsOnline[g] > 0:
            for t in range(m.InitialTime,  m.InitialTime + lg):
                m.UnitOn[g, t].fix(1)

        fg = min(m.FinalTime, int(m.InitialTimePeriodsOffline[g]))       # CarryOver Downtime
        if m.InitialTimePeriodsOffline[g] > 0: 
            for t in range(m.InitialTime, m.InitialTime + fg):
                m.UnitOn[g, t].fix(0)

    # m.MinUpTime_constraints   = ConstraintList(doc = 'MinUpTime_constraints')
    # m.MinDownTime_constraints = ConstraintList(doc = 'MinDownTime_constraints')
    
    # #Intra-window Uptime
    #     for t in range(m.InitialTime + lg, m.FinalTime + 1):
    #         kg = min(m.FinalTime - t + 1 , int(m.MinUpTime[g]))
    #         m.MinUpTime_constraints.add( sum(m.UnitOn[g,t] for t in range(t, t+kg)) >= kg * m.UnitStart[g,t] )

    # # # Intra-window Downtime
    #     for t in range(m.InitialTime + fg, m.FinalTime + 1):
    #         hg = min(m.FinalTime - t + 1, int(m.MinDownTime[g]))
    #         valid_tt = [tt for tt in range(t, t + hg) if tt in m.TimePeriods]
    #         m.MinDownTime_constraints.add( sum(m.UnitOn[g,tt] for tt in valid_tt) <= (1 - m.UnitStop[g,t]) * hg )
    
     # ======================================= Storage ======================================= #
     
    m.SoC_constraints = ConstraintList(doc="Storage")

    m.SoC_Under = Var(m.StorageUnits, within=NonNegativeReals) # slack variable: SoC shortfall below Init value 

    for b in m.StorageUnits:
        for t in m.TimePeriods:
            if t % 24 == 0:
                #m.SoC_constraints.add(m.SoC[b,t] >= m.SoCAtT0[b])  
                m.SoC_constraints.add(m.SoC_Under[b] >= m.SoCAtT0[b] - m.SoC[b, t])

            if t == m.InitialTime:
                m.SoC_constraints.add(
                    m.SoC[b,t] == m.SoCAtT0[b] + m.Storage_Efficiency[b] * m.ChargePower[b,t] - (m.DischargePower[b,t] / m.Storage_Efficiency[b]))
            else:
                m.SoC_constraints.add(
                    m.SoC[b,t] == m.SoC[b,t-1] + m.Storage_Efficiency[b] * m.ChargePower[b,t]- (m.DischargePower[b,t] / m.Storage_Efficiency[b]))

            m.SoC_constraints.add( m.ChargePower[b,t]   <= m.Storage_RoC[b] * m.IsCharging[b,t])
            m.SoC_constraints.add(m.DischargePower[b,t] <= m.Storage_RoC[b] * m.IsDischarging[b,t])
            m.SoC_constraints.add(m.IsCharging[b,t] + m.IsDischarging[b,t] <= 1)


 # ======================================= Load Solution for verification ======================================= #

    if fixed_commitment:
        print('fixing inputed variable values')
        for (g, t), val in fixed_commitment['UnitOn'].items():
            m.UnitOn[g, t].setlb(val)
            m.UnitOn[g, t].setub(val)
        for (g, t), val in fixed_commitment['UnitStart'].items():
            m.UnitStart[g, t].setlb(val)
            m.UnitStart[g, t].setub(val)
        for (g, t), val in fixed_commitment['UnitStop'].items():
            m.UnitStop[g, t].setlb(val)
            m.UnitStop[g, t].setub(val)
        for (b, t), val in fixed_commitment['IsCharging'].items():
            m.IsCharging[b, t].setlb(val)
            m.IsCharging[b, t].setub(val)
        for (b, t), val in fixed_commitment['IsDischarging'].items():
            m.IsDischarging[b, t].setlb(val)
            m.IsDischarging[b, t].setub(val)
        # for (b, t), val in fixed_commitment['SoC'].items():
        #     m.SoC[b, t].setlb(val)
        #     m.SoC[b, t].setub(val)
            
 # ======================================= Objective Function ======================================= #

    m.TimePrice = Param(m.TimePeriods, initialize=lambda m, t: 20.0 if (int(t) % 24) in (16, 17, 18, 19, 20) else 5.0)

    # Objective
    def ofv(m):
        start_cost = sum( m.StartUpCost[g] * m.UnitStart[g,t]              for g in m.ThermalGenerators   for t in m.TimePeriods)
        on_cost    = sum( m.CommitmentCost[g] * m.UnitOn[g,t]              for g in m.ThermalGenerators   for t in m.TimePeriods)
        power_cost = sum( data['gen_cost'][g]  * m.PowerGenerated[g,t]     for g in m.ThermalGenerators   for t in m.TimePeriods)
        renew_cost = sum( 0.01 * m.RenPowerGenerated[g,t]                  for g in m.RenewableGenerators for t in m.TimePeriods)
        disch_cost = sum( 20.0 * m.DischargePower[b,t]                     for b in m.StorageUnits        for t in m.TimePeriods)
        shed_cost  = sum( 1000 * m.LoadShed[n,t]                           for n in data["load_buses"]    for t in m.TimePeriods)

        #stop_cost  = sum(   m.ShutDownCost[g] * m.UnitStop[g,t]   for g in m.ThermalGenerators for t in m.TimePeriods)
        #c = sum(m.PowerCostVar[g,t] for g in m.ThermalGenerators for t in m.TimePeriods)
        return   start_cost + on_cost + power_cost + shed_cost + renew_cost + disch_cost + 5000 * sum(m.SoC_Under[b] for b in m.StorageUnits) 
        
        
    m.Objective = Objective(rule=ofv, sense=minimize)
    
 # ======================================= Solve ======================================= #
 
    build_time = perf_counter() - t0  # --------- stop BUILD timer 

    if fixed_commitment is None:
        print("build time monolithic:", build_time)

    opt.options['MIPGap'] = opt_gap

    t_solve = perf_counter()         # --------- start SOLVE timer

    results = opt.solve(m, tee = tee)

    solve_time = perf_counter() - t_solve

    mono_time = perf_counter() - t0
    
    if results.solver.termination_condition == TerminationCondition.optimal:
        if fixed_commitment is not None:
            print(f'Heuristic cost with stitched commitments:', round(value(m.Objective),2), '\n')
        else:
            print(f"Monolithic UC cost with opt gap {opt_gap} is:", round(value(m.Objective),2))
            print("Monolithic build + solve time:", round(mono_time,2))

    return_object = { 'ofv': value(m.Objective), 
                       'vars':  {'PowerGenerated': {(g,t): value(m.PowerGenerated[g,t]) for g in m.ThermalGenerators for t in range(m.InitialTime, m.FinalTime+1)}, 
                                         'UnitOn': {(g,t): value(m.UnitOn[g,t])         for g in m.ThermalGenerators for t in range(m.InitialTime, m.FinalTime+1)},
                                      'UnitStart': {(g,t): value(m.UnitStart[g,t])      for g in m.ThermalGenerators for t in range(m.InitialTime, m.FinalTime+1)},
                                       'UnitStop': {(g,t): value(m.UnitStop[g,t])       for g in m.ThermalGenerators for t in range(m.InitialTime, m.FinalTime+1)},
                                       'LoadShed': {(n,t): value(m.LoadShed[n,t])       for n in data['load_buses']  for t in range(m.InitialTime, m.FinalTime+1)},                                       
                                           'Flow': {(l,t): value(m.Flow[l,t])           for l in m.TransmissionLines for t in range(m.InitialTime, m.FinalTime+1)},
                                        'V_Angle': {(n,t): value(m.V_Angle[n,t])        for n in data['buses']       for t in range(m.InitialTime, m.FinalTime+1)}}}

    return_object['vars'].update({
        'SoC':            {(b,t): value(m.SoC[b,t])            for b in m.StorageUnits for t in range(m.InitialTime, m.FinalTime+1)},
        'ChargePower':    {(b,t): value(m.ChargePower[b,t])    for b in m.StorageUnits for t in range(m.InitialTime, m.FinalTime+1)},
        'DischargePower': {(b,t): value(m.DischargePower[b,t]) for b in m.StorageUnits for t in range(m.InitialTime, m.FinalTime+1)},
        'IsCharging':     {(b,t): value(m.IsCharging[b,t])     for b in m.StorageUnits for t in range(m.InitialTime, m.FinalTime+1)},
        'IsDischarging':  {(b,t): value(m.IsDischarging[b,t])  for b in m.StorageUnits for t in range(m.InitialTime, m.FinalTime+1)}})
    
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # --- Monolithic SoC plot ---

    # s = pd.Series(fixed_sol['SoC'])
    # s.index = pd.MultiIndex.from_tuples(s.index, names=['b', 't'])
    # df_soc = s.reorder_levels(['t', 'b']).sort_index().unstack('b')

    # out_dir = "RH_plots"
    # os.makedirs(out_dir, exist_ok=True)

    # Plot and save to folder
    # ax = df_soc.plot(figsize=(10, 6), linewidth=1.8, legend=False)
    # ax.set_xlabel("Time (t)")
    # ax.set_ylabel("SoC")
    # ax.set_title(f"SoC by battery (T={T}, Monolithic UC)")
    # plt.tight_layout()

    # plt.savefig(os.path.join(out_dir, f"SoC_T{T}_F{F}_L{L}.svg"), dpi=300)
    # plt.close()

    soc_mono = return_object['vars']['SoC']          
    s = pd.Series(soc_mono)
    s.index = pd.MultiIndex.from_tuples(s.index, names=['b','t'])
    df_soc = s.reorder_levels(['t','b']).sort_index().unstack('b')
    
    out_dir = "RH_plots_final"
    os.makedirs(out_dir, exist_ok=True)

    ax = df_soc.plot(figsize=(10, 6),linewidth=1.8,legend=False)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("SoC")
    ax.set_title(f"SoC by battery (T={T}, F={F}, L={L})")
    fig = ax.get_figure()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"RHSoC_T{T}_F{F}_L{L}.svg"), dpi=300)
    plt.close(fig)

    if save_sol:
        import csv
        all_t = sorted({t for t in m.TimePeriods})
        all_g = sorted({g for g in m.ThermalGenerators})
        all_b = sorted({b for b in m.StorageUnits})

        with open(f"Sol_bench{T}.csv", "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["Variable", "Entity"] + all_t)
            for g in all_g:
                row = ["UnitOn", g]
                for t in all_t:
                    row.append(return_object['vars']['UnitOn'].get((g, t), ""))
                writer.writerow(row)
            for g in all_g:
                row = ["UnitStart", g]
                for t in all_t:
                    row.append(return_object['vars']['UnitStart'].get((g, t), ""))
                writer.writerow(row)
            for g in all_g:
                row = ["UnitStop", g]
                for t in all_t:
                    row.append(return_object['vars']['UnitStop'].get((g, t), ""))
                writer.writerow(row)

            for b in all_b:
                row = ["SoC", b]
                for t in all_t:
                    row.append(return_object['vars']['SoC'].get((b, t), ""))
                writer.writerow(row)

            for b in all_b:
                row = ["ChargePower", b]
                for t in all_t:
                    row.append(return_object['vars']['ChargePower'].get((b, t), ""))
                writer.writerow(row)

            for b in all_b:
                row = ["DischargePower", b]
                for t in all_t:
                    row.append(return_object['vars']['DischargePower'].get((b, t), ""))
                writer.writerow(row)

            for b in all_b:
                row = ["IsCharging", b]
                for t in all_t:
                    row.append(return_object['vars']['IsCharging'].get((b, t), ""))
                writer.writerow(row)

            for b in all_b:
                row = ["IsDischarging", b]
                for t in all_t:
                    row.append(return_object['vars']['IsDischarging'].get((b, t), ""))
                writer.writerow(row)

    return return_object

        
#x = benchmark_UC_build(data, opt_gap=0.005, tee = True, save_sol = False)
