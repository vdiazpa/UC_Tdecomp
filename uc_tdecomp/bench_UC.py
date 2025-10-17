from uc_tdecomp.data_extract import load_uc_data, load_csv_data
import numpy as np
from pyomo.environ import *
from time import perf_counter

#enter filepath with data and subhorizon window size
file_path  = "examples/unit_commitment/RTS_GMLC_zonal_noreserves.json"
#file_path = "examples/unit_commitment/tiny_rts_ready.json"

data =  load_csv_data(36)
#data      = load_uc_data(file_path)


def benchmark_UC_build(data, opt_gap, fixed_commitment=None, seed=None):
    
    t0 = perf_counter()

    m = ConcreteModel()
    opt = SolverFactory('gurobi')
    opt.warm_start_capable()

    # Sets
    m.TimePeriods         = data['periods']
    m.LoadBuses           = Set(initialize=data['load_buses'], ordered=True)
    m.InitialTime         = min(m.TimePeriods)
    m.FinalTime           = max(m.TimePeriods)
    m.ThermalGenerators   = Set(initialize=data['ther_gens'], ordered=True)
    m.RenewableGenerators = Set(initialize=data['ren_gens'],  ordered=True)
    m.Generators          = Set(initialize=data['gens'],      ordered=True)
    m.TransmissionLines   = Set(initialize=data['lines'],     ordered=True)
    #m.CostSegments        = Set(initialize=range(1, data['n_segments']), ordered=True)  # number of piecewise cost segments

    W_full  = m.FinalTime - m.InitialTime + 1

    # Parameters 
    m.MinUpTime              = Param(m.ThermalGenerators, initialize = data['min_UT'])
    m.MinDownTime            = Param(m.ThermalGenerators, initialize = data['min_DT'])
    m.PowerGeneratedT0       = Param(m.ThermalGenerators, initialize = data['p_init'])   
    m.StatusAtT0             = Param(m.ThermalGenerators, initialize = data['init_status'] ) 
    m.UnitOnT0               = Param(m.ThermalGenerators, initialize = lambda m, g: 1.0 if m.StatusAtT0[g] > 0 else 0.0)
    
    m.InitialTimePeriodsOnline = Param(m.ThermalGenerators, 
                                        initialize = lambda m, g: (min(W_full, max(0, int(value(m.MinUpTime[g])) - int(value(m.StatusAtT0[g])))) if value(m.StatusAtT0[g]) > 0 else 0))
    
    m.InitialTimePeriodsOffline =  Param(m.ThermalGenerators, 
                                        initialize = lambda m, g: (min(W_full, max(0, int(value(m.MinDownTime[g])) - abs(int(value(m.StatusAtT0[g]))))) if value(m.StatusAtT0[g]) < 0 else 0))
    
    m.MaximumPowerOutput     = Param(m.ThermalGenerators,   initialize=data['p_max'])
    m.MinimumPowerOutput     = Param(m.ThermalGenerators,   initialize=data['p_min'])
    m.RenewableOutput        = Param(m.RenewableGenerators, data["periods"], initialize=data['ren_output'])
    m.CommitmentCost         = Param(m.ThermalGenerators,   initialize=data['commit_cost'])
    m.StartUpCost            = Param(m.ThermalGenerators,   initialize=data['startup_cost'])
    #m.ShutDownCost           = Param(m.ThermalGenerators,   initialize=data['shutdown_cost'])
    m.NominalRampUpLimit     = Param(m.ThermalGenerators,   initialize=data['rup'])
    m.NominalRampDownLimit   = Param(m.ThermalGenerators,   initialize=data['rdn'])
    m.StartupRampLimit       = Param(m.ThermalGenerators,   initialize=data['suR'])
    m.ShutdownRampLimit      = Param(m.ThermalGenerators,   initialize=data['sdR'])
    m.FlowCapacity           = Param(m.TransmissionLines,   initialize = data['line_cap'])
    m.LineReactance          = Param(m.TransmissionLines,   initialize = data['line_reac'])

    # Variables & Bounds
    m.PowerGenerated      = Var(m.ThermalGenerators, m.TimePeriods, within=NonNegativeReals) 
    m.PowerCostVar        = Var(m.ThermalGenerators, m.TimePeriods, within=NonNegativeReals) 
    m.UnitOn              = Var(m.ThermalGenerators, m.TimePeriods, within=Binary)
    m.UnitStart           = Var(m.ThermalGenerators, m.TimePeriods, within=Binary)
    m.UnitStop            = Var(m.ThermalGenerators, m.TimePeriods, within=Binary)
    m.V_Angle             = Var(  data['buses'],     m.TimePeriods, within = Reals, bounds = (-180, 180) )
    m.Flow                = Var(m.TransmissionLines, m.TimePeriods, within = Reals, bounds = lambda m, l, t: (-value(m.FlowCapacity[l]), value(m.FlowCapacity[l])))
    m.LoadShed            = Var(data["load_buses"], m.TimePeriods, within = NonNegativeReals)
    
    # Constraints
    m.MaxCapacity_thermal = Constraint(m.ThermalGenerators,  m.TimePeriods, rule=lambda m,g,t: m.PowerGenerated[g,t] <= m.MaximumPowerOutput[g]*m.UnitOn[g,t], doc= 'max_capacity_thermal')
    m.MinCapacity_thermal = Constraint(m.ThermalGenerators,  m.TimePeriods, rule=lambda m,g,t: m.PowerGenerated[g,t] >= m.MinimumPowerOutput[g]*m.UnitOn[g,t], doc= 'min_capacity_thermal')
    #m.MaxCapacity_renew   = Constraint(m.RenewableGenerators,m.TimePeriods, rule=lambda m,g,t: m.PowerGenerated[g,t] == m.RenewableOutput[(g,t)], doc= 'renewable_output')
    m.Power_Flow          = Constraint(m.TransmissionLines,  m.TimePeriods, rule=lambda m,l,t: m.Flow[l,t]*m.LineReactance[l] == m.V_Angle[data["line_ep"][l][0],t] - m.V_Angle[data["line_ep"][l][1],t], doc='Power_flow')

    def nb_rule(m,b,t):
        thermal = sum(m.PowerGenerated[g,t] for g in data["ther_gens_by_bus"][b]) if b in data["ther_gens_by_bus"] else 0.0
        flows   = sum(m.Flow[l,t] * data['lTb'][(l,b)] for l in data['lines_by_bus'][b])
        renew   = data['ren_bus_t'][(b,t)]
        shed    = m.LoadShed[b,t] if b in data["load_buses"] else 0.0
        return thermal + flows + renew + shed >= data["demand"].get((b,t), 0.0)
    
    m.NodalBalance = Constraint(data["buses"], m.TimePeriods, rule = nb_rule)

    for t in m.TimePeriods:
        m.V_Angle[data["ref_bus"], t].fix(0.0)

    m.logical_constraints  = ConstraintList(doc = 'logical')
    m.RampUp_constraints   = ConstraintList(doc = 'ramp_up')
    m.RampDown_constraints = ConstraintList(doc = 'ramp_down')
    
    # Add time-coupling constraints
    for g in m.ThermalGenerators:
        for t in m.TimePeriods: 
            if t == m.InitialTime:
                m.logical_constraints.add(m.UnitStart[g,t] - m.UnitStop[g,t] == m.UnitOn[g,t] - m.UnitOnT0[g])
                m.RampUp_constraints.add(m.PowerGenerated[g,t] - m.PowerGeneratedT0[g]<= m.NominalRampUpLimit[g] * m.UnitOnT0[g] + m.StartupRampLimit[g] * m.UnitStart[g,t])
                m.RampDown_constraints.add(m.PowerGeneratedT0[g] - m.PowerGenerated[g,t] <= m.NominalRampDownLimit[g] * m.UnitOn[g,t] + m.ShutdownRampLimit[g] * m.UnitStop[g,t]) #assumes power generated at 0 is 0

            else: 
                m.logical_constraints.add(m.UnitStart[g,t] - m.UnitStop[g,t] == m.UnitOn[g,t] - m.UnitOn[g,t-1])
                m.RampUp_constraints.add(m.PowerGenerated[g,t] - m.PowerGenerated[g,t-1] <= m.NominalRampUpLimit[g] * m.UnitOn[g, t-1] + m.StartupRampLimit[g] * m.UnitStart[g,t])
                m.RampDown_constraints.add(m.PowerGenerated[g,t-1] - m.PowerGenerated[g,t] <= m.NominalRampDownLimit[g] * m.UnitOn[g, t] + m.ShutdownRampLimit[g] * m.UnitStop[g,t])

    #UpTime Contraint Lists
    m.MinUpTime_constraints    = ConstraintList(doc='MinUpTime')
    m.MinDownTime_constraints  = ConstraintList(doc='MinDownTime')

    for g in m.ThermalGenerators:

    # CarryOver Uptime    
        lg = min(m.FinalTime, int(m.InitialTimePeriodsOnline[g]))
        if m.InitialTimePeriodsOnline[g] > 0:
            for t in range(m.InitialTime,  m.InitialTime + lg):
                m.UnitOn[g, t].fix(1)

    # CarryOver Downtime
        fg = min(m.FinalTime, int(m.InitialTimePeriodsOffline[g]))
        if m.InitialTimePeriodsOffline[g] > 0: 
            for t in range(m.InitialTime, m.InitialTime + fg):
                m.UnitOn[g, t].fix(0)

    # Intra-window Uptime
        for t in range(m.InitialTime + lg, m.FinalTime + 1):
            kg = min(m.FinalTime - t + 1 , int(m.MinUpTime[g]))
            m.MinUpTime_constraints.add( sum(m.UnitOn[g,t] for t in range(t, t+kg)) >= kg * m.UnitStart[g,t] )

    # # Intra-window Downtime
        for t in range(m.InitialTime + fg, m.FinalTime + 1):
            hg = min(m.FinalTime - t + 1, int(m.MinDownTime[g]))
            valid_tt = [tt for tt in range(t, t + hg) if tt in m.TimePeriods]
            m.MinDownTime_constraints.add( sum(m.UnitOn[g,tt] for tt in valid_tt) <= (1 - m.UnitStop[g,t]) * hg )

    
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
            
    # Objective
    def ofv(m):
        start_cost = sum( m.StartUpCost[g] * m.UnitStart[g,t]  for g in m.ThermalGenerators   for t in m.TimePeriods)
        on_cost    = sum( m.CommitmentCost[g] * m.UnitOn[g,t]  for g in m.ThermalGenerators   for t in m.TimePeriods)
        power_cost = sum( 10  * m.PowerGenerated[g,t]          for g in m.ThermalGenerators   for t in m.TimePeriods)
        shed_cost  = sum( 1000 * m.LoadShed[n,t]               for n in data["load_buses"]     for t in m.TimePeriods)
        #stop_cost  = sum(   m.ShutDownCost[g] * m.UnitStop[g,t]   for g in m.ThermalGenerators for t in m.TimePeriods)
        #c = sum(m.PowerCostVar[g,t] for g in m.ThermalGenerators for t in m.TimePeriods)
        return   start_cost + on_cost + power_cost + shed_cost
        
    m.Objective = Objective(rule=ofv, sense=minimize)

    build_time = perf_counter() - t0 # --------- stop BUILD timer 

    if fixed_commitment is None:
        print("build time monolithic:", build_time)

    #opt = SolverFactory('gurobi')
    if seed: 
        opt.options['Seed'] = seed
    opt.options['MIPGap'] = opt_gap
    #opt.options['Heuristics'] = 0.3

    t_solve = perf_counter()         # --------- start SOLVE timer

    results = opt.solve(m, tee = False)

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

    # for (g, t), v in sorted(return_object['vars']['UnitOn'].items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
    #     if abs(v) > 1e-6:
    #         print(f"UnitOn[{g},{t}] = {int(round(v))}")

    import csv

    # total_ls = sum(float(v) for v in return_object['vars']['LoadShed'].values())
    # print(f'Total load shed cost: {total_ls * 1000}')

    return return_object

        
#x = benchmark_UC_build(data, opt_gap=0.1)
