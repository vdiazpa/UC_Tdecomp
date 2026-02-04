#bench_UC.py
from pyomo.environ import *
from time import perf_counter
import math

def benchmark_UC_build(data, save_sol_to:str = False, opt_gap=0.01, fixed_commitment=None, 
        tee=False, do_solve=True, MUT= "classical", MDT= "classical"):
    """Build full horizon UC model & solve.

    Parameters
    ----------
    data : dict
        Dictionary with model objects (network & unit parameters, maps, etc.)
    save_sol_to : str , optional
        Name of file if wish to save solution. 
    fixed_commitment : dict
        Dictionary with (Bool) solution to fix var values to.
    Returns
    -------
    dict
        {'ofv': Float, 'vars': Dict with var values}
    """
    m   = ConcreteModel()
    t0  = perf_counter() if do_solve else None

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
    # m.MinUpTime          = Param(m.ThermalGenerators, initialize = data['min_UT'])
    # m.MinDownTime        = Param(m.ThermalGenerators, initialize = data['min_DT'])
    m.MinUpTime        = Param(m.ThermalGenerators, initialize=lambda m,g: int(math.ceil(data['min_UT'][g])))
    m.MinDownTime      = Param(m.ThermalGenerators, initialize=lambda m,g: int(math.ceil(data['min_DT'][g])))
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
    #m.PowerCostVar        = Var(m.ThermalGenerators, m.TimePeriods, within=NonNegativeReals) 
    # m.UTRemain            = Var(m.ThermalGenerators, m.TimePeriods, within=NonNegativeReals) 
    # m.DTRemain            = Var(m.ThermalGenerators, m.TimePeriods, within=NonNegativeReals) 
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
    # m.UTRemain_constraints = ConstraintList(doc = 'UT_remain')
    # m.DTRemain_constraints = ConstraintList(doc = 'DT_remain')

    for g in m.ThermalGenerators:
        for t in m.TimePeriods: 
            # m.UTRemain_constraints.add( m.UTRemain[g,t] <= m.MinUpTime[g] * m.UnitOn[g,t]) 
            # m.DTRemain_constraints.add( m.DTRemain[g,t] <= m.MinDownTime[g] * (1 - m.UnitOn[g,t]))
            
            
            if t == m.InitialTime:
                m.logical_constraints.add(m.UnitStart[g,t] - m.UnitStop[g,t] == m.UnitOn[g,t] - m.UnitOnT0[g])
                m.RampUp_constraints.add(m.PowerGenerated[g,t] - m.PowerGeneratedT0[g]<= m.NominalRampUpLimit[g] * m.UnitOnT0[g] + m.StartupRampLimit[g] * m.UnitStart[g,t])
                m.RampDown_constraints.add(m.PowerGeneratedT0[g] - m.PowerGenerated[g,t] <= m.NominalRampDownLimit[g] * m.UnitOn[g,t] + m.ShutdownRampLimit[g] * m.UnitStop[g,t]) #assumes power generated at 0 is 0
                
                # m.UTRemain_constraints.add( m.UTRemain[g,t] >=  m.InitialTimePeriodsOnline[g] +m.MinUpTime[g]*m.UnitStart[g,t] - m.UnitOn[g,t])
                # m.DTRemain_constraints.add( m.DTRemain[g,t] >=  m.InitialTimePeriodsOffline[g] +m.MinDownTime[g]*m.UnitStop[g,t] - (1 -m.UnitOn[g,t]))

            else: 
                m.logical_constraints.add(m.UnitStart[g,t] - m.UnitStop[g,t] == m.UnitOn[g,t] - m.UnitOn[g,t-1])
                m.RampUp_constraints.add(m.PowerGenerated[g,t] - m.PowerGenerated[g,t-1] <= m.NominalRampUpLimit[g] * m.UnitOn[g, t-1] + m.StartupRampLimit[g] * m.UnitStart[g,t])
                m.RampDown_constraints.add(m.PowerGenerated[g,t-1] - m.PowerGenerated[g,t] <= m.NominalRampDownLimit[g] * m.UnitOn[g, t] + m.ShutdownRampLimit[g] * m.UnitStop[g,t])
                
                # m.UTRemain_constraints.add( m.UTRemain[g,t] >=  m.UTRemain[g,t-1] +m.MinUpTime[g]*m.UnitStart[g,t] - m.UnitOn[g,t])
                # m.DTRemain_constraints.add( m.DTRemain[g,t] >=  m.DTRemain[g,t-1] +m.MinDownTime[g]*m.UnitStop[g,t] - (1 -m.UnitOn[g,t]))



# ======================================= Minimum Up/Down Time ======================================= #


#     for g in m.ThermalGenerators:
        
#         lg = min(m.FinalTime, int(m.InitialTimePeriodsOnline[g]))         # CarryOver Uptime    
#         if m.InitialTimePeriodsOnline[g] > 0:
#             for t in range(m.InitialTime,  m.InitialTime + lg):
#                 m.UnitOn[g, t].fix(1)
        # Carryover downtime
        
        fg = min(m.FinalTime, int(m.InitialTimePeriodsOffline[g]))
        if m.InitialTimePeriodsOffline[g] > 0:
            for t in range(m.InitialTime, m.InitialTime + fg):
                m.UnitOn[g, t].fix(0)

        # Intra-horizon MUT
        for t in range(m.InitialTime + lg, m.FinalTime + 1):
            kg = min(m.FinalTime - t + 1, int(m.MinUpTime[g]))
            m.MinUpTime_constraints.add(
                sum(m.UnitOn[g, tt] for tt in range(t, t + kg)) >= kg * m.UnitStart[g, t])

        # Intra-horizon MDT
        for t in range(m.InitialTime + fg, m.FinalTime + 1):
            hg = min(m.FinalTime - t + 1, int(m.MinDownTime[g]))
            valid_tt = [tt for tt in range(t, t + hg) if tt in m.TimePeriods]
            m.MinDownTime_constraints.add(
                sum(m.UnitOn[g, tt] for tt in valid_tt) <= (1 - m.UnitStop[g, t]) * hg)
            
    if MUT == "classical" and MDT == "classical":
        m.MinUpTime_constraints    = ConstraintList(doc='MinUpTime')
        m.MinDownTime_constraints  = ConstraintList(doc='MinDownTime')

        for g in m.ThermalGenerators:
            lg = min(W_full, int(value(m.InitialTimePeriodsOnline[g])))
            fg = min(W_full, int(value(m.InitialTimePeriodsOffline[g])))

            # Carryover commitments from initial status
            if lg > 0:
                for t in range(m.InitialTime, m.InitialTime + lg):
                    m.UnitOn[g, t].fix(1)
            if fg > 0:
                for t in range(m.InitialTime, m.InitialTime + fg):
                    m.UnitOn[g, t].fix(0)

            # Intra-horizon MUT
            for t in range(m.InitialTime + lg, m.FinalTime + 1):
                kg = min(m.FinalTime - t + 1, int(value(m.MinUpTime[g])))
                m.MinUpTime_constraints.add(sum(m.UnitOn[g, tt] for tt in range(t, t + kg)) >= kg * m.UnitStart[g, t])

            # Intra-horizon MDT
            for t in range(m.InitialTime + fg, m.FinalTime + 1):
                hg = min(m.FinalTime - t + 1, int(value(m.MinDownTime[g])))
                m.MinDownTime_constraints.add(sum((1 - m.UnitOn[g, tt]) for tt in range(t, t + hg)) >= hg * m.UnitStop[g, t])

    elif MUT == "counter" and MDT == "counter":
        m.UTRemain = Var(m.ThermalGenerators, m.TimePeriods, within=NonNegativeReals)
        m.DTRemain = Var(m.ThermalGenerators, m.TimePeriods, within=NonNegativeReals)
        m.UTRemain_constraints = ConstraintList()
        m.DTRemain_constraints = ConstraintList()

        for g in m.ThermalGenerators:
            for t in m.TimePeriods:
                m.UTRemain_constraints.add(m.UTRemain[g, t] <= m.MinUpTime[g] * m.UnitOn[g, t])
                m.DTRemain_constraints.add(m.DTRemain[g, t] <= m.MinDownTime[g] * (1 - m.UnitOn[g, t]))

                if t == m.InitialTime:
                    m.UTRemain_constraints.add(m.UTRemain[g, t] >= m.InitialTimePeriodsOnline[g] + m.MinUpTime[g] * m.UnitStart[g, t] - m.UnitOn[g, t])
                    m.DTRemain_constraints.add(
                        m.DTRemain[g, t] >= m.InitialTimePeriodsOffline[g] + m.MinDownTime[g] * m.UnitStop[g, t] - (1 - m.UnitOn[g, t]))
                else:
                    m.UTRemain_constraints.add(m.UTRemain[g, t] >= m.UTRemain[g, t-1] + m.MinUpTime[g] * m.UnitStart[g, t] - m.UnitOn[g, t])
                    m.DTRemain_constraints.add(m.DTRemain[g, t] >= m.DTRemain[g, t-1] + m.MinDownTime[g] * m.UnitStop[g, t] - (1 - m.UnitOn[g, t]))
    else:
        raise ValueError(f"Unsupported MUT/MDT selection: MUT={MUT}, MDT={MDT}")


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
            
 # ======================================= Objective Function ======================================= #

    m.StageOneCost = Expression(expr = 
        sum( m.StartUpCost[g] * m.UnitStart[g,t]     for g in m.ThermalGenerators   for t in m.TimePeriods)
        + sum( m.CommitmentCost[g] * m.UnitOn[g,t]   for g in m.ThermalGenerators   for t in m.TimePeriods) )

    # Objective
    def ofv(m):
        start_cost = sum( m.StartUpCost[g] * m.UnitStart[g,t]              for g in m.ThermalGenerators   for t in m.TimePeriods)
        on_cost    = sum( m.CommitmentCost[g] * m.UnitOn[g,t]              for g in m.ThermalGenerators   for t in m.TimePeriods)
        power_cost = sum( data['gen_cost'][g]  * m.PowerGenerated[g,t]     for g in m.ThermalGenerators   for t in m.TimePeriods)
        renew_cost = sum( 0.01 * m.RenPowerGenerated[g,t]                  for g in m.RenewableGenerators for t in m.TimePeriods)
        disch_cost = sum( 20.0 * m.DischargePower[b,t]                     for b in m.StorageUnits        for t in m.TimePeriods)
        ch_cost = sum( 20.0 * m.ChargePower[b,t]                for b in m.StorageUnits        for t in m.TimePeriods)
        shed_cost  = sum( 1000 * m.LoadShed[n,t]                           for n in data["load_buses"]    for t in m.TimePeriods)

        return   start_cost + on_cost + power_cost + shed_cost + renew_cost + disch_cost + ch_cost + 5000 * sum(m.SoC_Under[b] for b in m.StorageUnits) 
        
        
    m.Objective = Objective(rule=ofv, sense=minimize)

    if not do_solve: 
        return m
    
 # ======================================= Solve ======================================= #
 
    build_time = perf_counter() - t0  # --------- stop BUILD timer 

    if fixed_commitment is None:
        print("build time monolithic:", build_time)

    opt = SolverFactory('gurobi')
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
    

    if save_sol_to:
        import csv
        all_t = sorted({t for t in m.TimePeriods})
        all_g = sorted({g for g in m.ThermalGenerators})
        all_b = sorted({b for b in m.StorageUnits})

        with open(save_sol_to, "w", newline="") as f:
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


def plot_soc_from_return(return_object, out_dir='SoC Plots', prefix=None, T=None, F=None, L=None, *, save_svg=True, save_pdf=True, dpi=300, figsize=(3.5, 1.5), show=False):
    """Plot state-of-charge (SoC) time series for all storage units.

    Parameters
    ----------
    return_object : dict
        Dictionary returned by this function containing `vars` -> `SoC` mapping {(b,t): value}.
    out_dir : str or Path, optional
        Directory to save plots (created if missing). Default 'SoC Plots'.
    prefix : str, optional
        Optional filename prefix.
    T, F, L : any, optional
        Identifiers used in filenames (kept generic since they may not be available here).
    save_svg, save_pdf : bool
        Whether to save SVG/PDF outputs.
    dpi : int
        DPI for saved figures.
    figsize : tuple
        Figure size in inches.
    show : bool
        If True, call `plt.show()` after plotting.

    Returns
    -------
    dict
        {'df_soc': DataFrame, 'files': [saved_file_paths]}
    """
    import os
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    soc = return_object.get('vars', {}).get('SoC')
    if soc is None or len(soc) == 0:
        raise ValueError('return_object does not contain any SoC data')

    s = pd.Series(soc)
    s.index = pd.MultiIndex.from_tuples(s.index, names=['b', 't'])
    df_soc = s.reorder_levels(['t', 'b']).sort_index().unstack('b')

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []

    # simple pandas plot (SVG)
    if save_svg:
        ax = df_soc.plot(figsize=figsize, linewidth=1.8, legend=False)
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('SoC')
        title_parts = []
        if T is not None:
            title_parts.append(f'T={T}')
        if F is not None:
            title_parts.append(f'F={F}')
        if L is not None:
            title_parts.append(f'L={L}')
        if title_parts:
            ax.set_title('SoC by battery (' + ', '.join(title_parts) + ')')
        fig = ax.get_figure()
        filename = (prefix + '_' if prefix else '') + f"RHSoC_T{T}_F{F}_L{L}.svg"
        path = out_dir / filename
        plt.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        files.append(str(path))

    # publication-style PDF plot
    if save_pdf:
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams.update({'font.family': 'Times New Roman'})
        colors = plt.cm.tab20.colors * 10
        for i, col in enumerate(df_soc.columns):
            ax.plot(range(len(df_soc)), df_soc[col], linewidth=0.8, color=colors[i], label=f'ESS {i+1}')
        ax.set_xlabel('Time (hr)', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.set_ylabel('ESS SoC(MWh)', fontsize=8)
        ax.grid(False)
        plt.tight_layout()
        filename = (prefix + '_' if prefix else '') + f"RHSoC_T{T}_F{F}_L{L}.pdf"
        path = out_dir / filename
        fig.savefig(path, pad_inches=0.01, bbox_inches='tight', dpi=600)
        plt.close(fig)
        files.append(str(path))

    if show:
        import matplotlib.pyplot as _plt
        _plt.show()

    return {'df_soc': df_soc, 'files': files}

# Example usage:
# plot_soc_from_return(return_object, out_dir='RH_plots_final', prefix='RHSoC', T=T, F=F, L=L)

def sweep_benchmark(data, T_vals=(24, 72, 168), opt_gaps=(0.001,), solver_name='gurobi', only_valid=False, csv_path=None, verbose=False):
    """Sweep benchmark over multiple horizons (T) and optimality gaps.

    This function creates time-sliced copies of `data` (keeping only the earliest
    T periods) and runs `benchmark_UC_build` for each combination of T and opt-gap.
    It records runtime, objective value, success status and any error message, and
    saves the results to `csv_path` if provided.

    Parameters
    ----------
    data : dict
        Full data dictionary (contains 'periods' and time-indexed dicts like 'ren_output' and 'demand').
    T_vals : iterable of int
        List/tuple of horizons (number of periods) to evaluate.
    opt_gaps : iterable of float
        List/tuple of MIP gaps to test.
    solver_name : str
        Name of solver for metadata (not currently used to construct solver instance here).
    only_valid : bool
        If True, skip T > len(data['periods']).
    csv_path : str or Path, optional
        Path to write CSV summary. If None, a default name will be generated and written to CWD.
    verbose : bool
        Print progress and failures.

    Returns
    -------
    pandas.DataFrame
        Summary dataframe with one row per (T, opt_gap) run.
    """
    import copy
    import time
    import datetime
    from pathlib import Path
    import numpy as np
    import pandas as pd

    original_periods = sorted(list(data.get('periods', [])))
    if len(original_periods) == 0:
        raise ValueError("`data` must contain a non-empty 'periods' sequence")

    # normalize single-value opt_gaps
    if isinstance(opt_gaps, (float, int)):
        opt_gaps = (float(opt_gaps),)

    records = []

    for T in T_vals:
        if T > len(original_periods):
            if only_valid:
                if verbose:
                    print(f"Skipping T={T} (only {len(original_periods)} periods available)")
                continue
            else:
                if verbose:
                    print(f"Warning: requested T={T} exceeds available periods ({len(original_periods)}); truncating")
                use_periods = original_periods
        else:
            use_periods = original_periods[:T]

        # make a deepcopy and keep only time entries for periods in use_periods
        data_sub = copy.deepcopy(data)
        data_sub['periods'] = use_periods

        # Filter time-indexed dicts: keep non-tuple-key dicts and tuple-key dicts where key[1] in use_periods
        for k, v in list(data_sub.items()):
            if isinstance(v, dict):
                # detect if dict is time-indexed by checking for tuple-like keys
                if any(isinstance(k2, tuple) and len(k2) >= 2 and k2[1] in original_periods for k2 in v.keys()):
                    filtered = {k2: val for k2, val in v.items() if not (isinstance(k2, tuple) and len(k2) >= 2 and k2[1] not in use_periods)}
                    data_sub[k] = filtered

        for opt_gap in opt_gaps:
            if verbose:
                print(f"Running benchmark: T={T}, opt_gap={opt_gap}")

            t0 = time.perf_counter()
            try:
                # Do not save UC solution by default
                ret = benchmark_UC_build(data_sub, save_sol_to=None, opt_gap=opt_gap, fixed_commitment=None, tee=False)
                runtime = time.perf_counter() - t0
                ofv = ret.get('ofv', np.nan) if isinstance(ret, dict) else np.nan
                success = True
                error = None
            except Exception as e:
                runtime = time.perf_counter() - t0
                ofv = np.nan
                success = False
                error = str(e)
                if verbose:
                    print(f"  failed (T={T}, opt_gap={opt_gap}): {e}")

            rec = {
                'run_id': f"T{T}_g{opt_gap}_{int(time.time()*1000) % 1000000}",
                'T': T,
                'opt_gap': opt_gap,
                'solver': solver_name,
                'runtime_sec': runtime,
                'ofv': ofv,
                'success': success,
                'error': error,
                'timestamp': datetime.datetime.now().isoformat()
            }
            records.append(rec)

    df = pd.DataFrame.from_records(records)

    if csv_path is None:
        csv_path = Path(f"bench_sweep_T{min(T_vals)}-{max(T_vals)}_{int(time.time())}.csv")
    else:
        csv_path = Path(csv_path)

    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"Wrote {len(df)} rows to {csv_path}")

    return df