
from pyomo.environ import *
import numpy as np
#from time import perf_counter

def build_RH_subprobs(data, s_e, init_state, fixed, print_carryover = False, opt_gap=0.01, warm_start = None, solver_seed=None, next_fixed_len=0):
    
    #t0 = perf_counter()
    m = ConcreteModel()

    t_fix0, t_fix1 = fixed
    F_k = t_fix1 - t_fix0 + 1   
    
    # Sets
    m.TimePeriods         = s_e
    m.LoadBuses           = Set(initialize=data['load_buses'], ordered=True)
    W = len(s_e)
    m.InitialTime         = min(s_e) #t0
    m.FinalTime           = max(s_e) #te
    m.ThermalGenerators   = Set(initialize=data['ther_gens'], ordered=True)
    m.RenewableGenerators = Set(initialize=data['ren_gens'],  ordered=True)
    m.Generators          = Set(initialize=data['gens'],      ordered=True)
    m.TransmissionLines   = Set(initialize=data['lines'],     ordered = True)
    m.StorageUnits        = Set(initialize = data['bats'],       ordered=True)
    #m.CostSegments        = Set(initialize=range(1, data['n_segments']), ordered=True)  # number of piecewise cost segments

    # Parameters 
    m.MinUpTime        = Param(m.ThermalGenerators, initialize = data['min_UT'])
    m.MinDownTime      = Param(m.ThermalGenerators, initialize = data['min_DT'])
    m.PowerGeneratedT0 = Param(m.ThermalGenerators, initialize = lambda m, g:(init_state.get('PowerGeneratedT0',{}).get(g,data['p_init'][g])) )   
    m.StatusAtT0       = Param(m.ThermalGenerators, initialize = lambda m, g:(init_state.get('StatusAtT0',      {}).get(g,data['init_status'][g])) ) 
    m.SoCAtT0          = Param(m.StorageUnits,      initialize = lambda m, b:(init_state.get('SoCT0',           {}).get(b,data['SoC_init'][b])) ) 

    m.UnitOnT0         = Param(m.ThermalGenerators, initialize = lambda m, g: 1.0 if m.StatusAtT0[g] > 0 else 0.0)
    
    m.InitialTimePeriodsOnline = Param(m.ThermalGenerators, 
                                        initialize = lambda m, g: (min(W, max(0, int(value(m.MinUpTime[g])) - int(value(m.StatusAtT0[g])))) if value(m.StatusAtT0[g]) > 0 else 0))
    
    m.InitialTimePeriodsOffline =  Param(m.ThermalGenerators, 
                                        initialize = lambda m, g: min(W, max(0, int(value(m.MinDownTime[g])) - abs(int(value(m.StatusAtT0[g])))) if value(m.StatusAtT0[g]) < 0 else 0))

    m.MaximumPowerOutput    = Param(m.ThermalGenerators, initialize = data['p_max'])
    m.MinimumPowerOutput    = Param(m.ThermalGenerators, initialize = data['p_min'])
    m.RenewableOutput       = Param(m.RenewableGenerators, data["periods"], initialize=data['ren_output'])
    m.CommitmentCost        = Param(m.ThermalGenerators, initialize = data['commit_cost'])
    m.StartUpCost           = Param(m.ThermalGenerators, initialize = data['startup_cost'])
    #m.ShutDownCost           = Param(m.ThermalGenerators,   initialize=data['shutdown_cost'])
    m.NominalRampUpLimit    = Param(m.ThermalGenerators, initialize = data['rup'])
    m.NominalRampDownLimit  = Param(m.ThermalGenerators, initialize = data['rdn'])
    m.StartupRampLimit      = Param(m.ThermalGenerators, initialize = data['suR'])
    m.ShutdownRampLimit     = Param(m.ThermalGenerators, initialize = data['sdR'])
    m.FlowCapacity          = Param(m.TransmissionLines, initialize = data['line_cap'])
    m.LineReactance         = Param(m.TransmissionLines, initialize = data['line_reac'])

    m.Storage_RoC         = Param(m.StorageUnits, initialize = data['sto_RoC'])
    m.Storage_EnergyCap   = Param(m.StorageUnits, initialize = data['sto_Ecap'])
    m.Storage_Efficiency  = Param(m.StorageUnits, initialize = data['sto_eff'])


    # Variables & Bounds
    m.PowerGenerated      = Var(m.ThermalGenerators,  m.TimePeriods, within = NonNegativeReals )#bounds = lambda m, g, t: (0, m.MaximumPowerOutput[g]))
    m.RenPowerGenerated   = Var(m.RenewableGenerators, m.TimePeriods, within=NonNegativeReals) 
    #m.PowerCostVar        = Var(m.ThermalGenerators, m.TimePeriods, within = NonNegativeReals) 
    m.UnitOn              = Var(m.ThermalGenerators, m.TimePeriods, within = Binary)
    m.UnitStart           = Var(m.ThermalGenerators, m.TimePeriods, within = Binary)
    m.UnitStop            = Var(m.ThermalGenerators, m.TimePeriods, within = Binary)
    m.V_Angle             = Var(  data['buses'],     m.TimePeriods, within = Reals, bounds = (-180, 180) )
    m.Flow                = Var(m.TransmissionLines, m.TimePeriods, within = Reals, bounds = lambda m, l, t: (-m.FlowCapacity[l], m.FlowCapacity[l]))
    m.LoadShed            = Var(data["load_buses"],  m.TimePeriods, within = NonNegativeReals)
    m.SoC                 = Var(m.StorageUnits,      m.TimePeriods, within = NonNegativeReals, bounds = lambda m, b, t: (0, m.Storage_EnergyCap[b]) )
    m.ChargePower         = Var(m.StorageUnits,      m.TimePeriods, within = NonNegativeReals, bounds = lambda m, b, t: (0, m.Storage_RoC[b]) )
    m.DischargePower      = Var(m.StorageUnits,      m.TimePeriods, within = NonNegativeReals, bounds = lambda m, b, t: (0, m.Storage_RoC[b]) )
    m.IsCharging          = Var(m.StorageUnits,      m.TimePeriods, within = Binary)
    m.IsDischarging       = Var(m.StorageUnits,      m.TimePeriods, within = Binary)

    # ======================================= Single-period constraints ======================================= #

    m.MaxCap_thermal     = Constraint(m.ThermalGenerators,  m.TimePeriods, rule=lambda m,g,t: m.PowerGenerated[g,t] <= m.MaximumPowerOutput[g]*m.UnitOn[g,t], doc= 'max_capacity_thermal')
    m.MinCap_thermal     = Constraint(m.ThermalGenerators,  m.TimePeriods, rule=lambda m,g,t: m.PowerGenerated[g,t] >= m.MinimumPowerOutput[g]*m.UnitOn[g,t], doc= 'min_capacity_thermal')
    m.Power_Flow         = Constraint(m.TransmissionLines,  m.TimePeriods, rule=lambda m,l,t: m.Flow[l,t]*m.LineReactance[l]==m.V_Angle[data["line_ep"][l][0],t]-m.V_Angle[data["line_ep"][l][1],t], doc='Power_flow')
    m.MaxCapacity_renew  = Constraint(m.RenewableGenerators, m.TimePeriods, rule=lambda m,g,t: m.RenPowerGenerated[g,t] <= m.RenewableOutput[(g,t)], doc= 'renewable_output')

    def nb_rule(m,b,t):
        thermal = sum(m.PowerGenerated[g,t] for g in data["ther_gens_by_bus"][b]) if b in data["ther_gens_by_bus"] else 0.0
        flows   = sum(m.Flow[l,t] * data['lTb'][(l,b)] for l in data['lines_by_bus'][b])
        renew   = 0.0 if b not in data['bus_ren_dict'] else sum(m.RenPowerGenerated[g,t] for g in data['bus_ren_dict'][b])
        shed    = m.LoadShed[b,t] if b in data["load_buses"] else 0.0
        storage = 0.0 if b not in data['bus_bat'] else sum(m.DischargePower[bat,t] - m.ChargePower[bat,t] for bat in data['bus_bat'][b])
        return thermal + flows + renew + shed + storage == data["demand"].get((b,t), 0.0)
    
    m.NodalBalance = Constraint(data["buses"], m.TimePeriods , rule = nb_rule) #range(m.InitialTime, t_fix1+1)
    
    for t in m.TimePeriods:
        m.V_Angle[data["ref_bus"], t].fix(0.0)

    # ====================================== Battery Energy Storage Constraints ====================================== #
    
    m.Storage_constraints = ConstraintList(doc='SoC_constraints')

    for b in m.StorageUnits:
        for t in m.TimePeriods:
            if t == m.InitialTime:
                m.Storage_constraints.add(m.SoC[b, t] == m.SoCAtT0[b] + m.Storage_Efficiency[b] * m.ChargePower[b, t]- m.DischargePower[b, t] / m.Storage_Efficiency[b])
            else:
                m.Storage_constraints.add(m.SoC[b, t] == m.SoC[b, t-1]+ m.Storage_Efficiency[b] * m.ChargePower[b, t]- m.DischargePower[b, t] / m.Storage_Efficiency[b])

            if t <= t_fix1:
                m.Storage_constraints.add(m.IsCharging[b, t] + m.IsDischarging[b, t] <= 1)
                m.Storage_constraints.add(m.ChargePower[b, t]  <= m.Storage_RoC[b] * m.IsCharging[b, t])
                m.Storage_constraints.add(m.DischargePower[b, t] <= m.Storage_RoC[b] * m.IsDischarging[b, t])
            else:
                m.IsCharging[b,t].domain    = UnitInterval                   # relax integrality after t_fix1
                m.IsDischarging[b,t].domain = UnitInterval
                m.Storage_constraints.add(m.IsCharging[b, t] + m.IsDischarging[b, t] <= 1)
                m.Storage_constraints.add(m.ChargePower[b, t]  <= m.Storage_RoC[b] * m.IsCharging[b, t])
                m.Storage_constraints.add(m.DischargePower[b, t] <= m.Storage_RoC[b] * m.IsDischarging[b, t])

    for b in m.StorageUnits:
        
        gain_next = next_fixed_len * m.Storage_Efficiency[b] * m.Storage_RoC[b]
        
        if gain_next >= m.SoCAtT0[b]:                   # if next window has plenty of time: enforce full neutrality now
            lb = m.SoCAtT0[b]
        else:
            lb = max(0.0, m.SoCAtT0[b] - gain_next)     # if not enough time next window: enforce  “reachable target” lower bound now

        m.Storage_constraints.add(m.SoC[b, t_fix1] >= lb) 
                   
    # # ======================================= Ramping & Logical Constraints ======================================= #
    
    m.RampDown_constraints = ConstraintList(doc = 'ramp_dn')
    m.logical_constraints  = ConstraintList(doc = 'logical')
    m.RampUp_constraints   = ConstraintList(doc = 'ramp_up')     
    
    if t_fix1 < m.FinalTime: 
        for g in m.ThermalGenerators:
            for t in m.TimePeriods:
                if t <= t_fix1:
                    if t == m.InitialTime:
                        m.logical_constraints.add(m.UnitStart[g,t] - m.UnitStop[g,t] == m.UnitOn[g,t] - m.UnitOnT0[g])
                        m.RampUp_constraints.add(m.PowerGenerated[g,t] - m.PowerGeneratedT0[g]<= m.NominalRampUpLimit[g] * m.UnitOnT0[g] + m.StartupRampLimit[g] * m.UnitStart[g,t])
                        m.RampDown_constraints.add(m.PowerGeneratedT0[g] - m.PowerGenerated[g,t]<= m.NominalRampDownLimit[g] * m.UnitOn[g,t] + m.ShutdownRampLimit[g] * m.UnitStop[g,t]) 
                    else:
                        m.logical_constraints.add(m.UnitStart[g,t] - m.UnitStop[g,t] == m.UnitOn[g,t] - m.UnitOn[g,t-1])
                        m.RampUp_constraints.add(m.PowerGenerated[g,t] - m.PowerGenerated[g,t-1]<= m.NominalRampUpLimit[g] * m.UnitOn[g,t-1] + m.StartupRampLimit[g] * m.UnitStart[g,t])
                        m.RampDown_constraints.add(m.PowerGenerated[g,t-1] - m.PowerGenerated[g,t]<= m.NominalRampDownLimit[g] * m.UnitOn[g,t] + m.ShutdownRampLimit[g] * m.UnitStop[g,t])
                else:
                    #print("Relaxing unit commitment var for ", g, " at t = ", t, "will also relax some constraints")
                    m.UnitOn[g,t].domain = UnitInterval
                    m.UnitStart[g,t].fix(0)
                    m.UnitStop[g,t].fix(0)

                    if t != m.InitialTime:
                        m.RampUp_constraints.add(m.PowerGenerated[g,t] - m.PowerGenerated[g,t-1]<= m.NominalRampUpLimit[g])     
                        m.RampDown_constraints.add(m.PowerGenerated[g,t-1] - m.PowerGenerated[g,t]<= m.NominalRampDownLimit[g])
    else: 
        for g in m.ThermalGenerators:
            for t in m.TimePeriods: 
                if t == m.InitialTime:
                    m.logical_constraints.add(m.UnitStart[g,t] - m.UnitStop[g,t] == m.UnitOn[g,t] - m.UnitOnT0[g])
                    m.RampUp_constraints.add(m.PowerGenerated[g,t] - m.PowerGeneratedT0[g]<= m.NominalRampUpLimit[g] * m.UnitOnT0[g] + m.StartupRampLimit[g] * m.UnitStart[g,t])
                    m.RampDown_constraints.add(m.PowerGeneratedT0[g] - m.PowerGenerated[g,t] <= m.NominalRampDownLimit[g] * m.UnitOn[g, t] + m.ShutdownRampLimit[g] * m.UnitStop[g,t]) 

                else:
                    m.logical_constraints.add(m.UnitStart[g,t] - m.UnitStop[g,t] == m.UnitOn[g,t] - m.UnitOn[g,t-1])
                    m.RampUp_constraints.add(  m.PowerGenerated[g,t] - m.PowerGenerated[g,t-1] <= m.NominalRampUpLimit[g] * m.UnitOn[g,t-1] + m.StartupRampLimit[g]  * m.UnitStart[g,t])
                    m.RampDown_constraints.add(m.PowerGenerated[g,t-1] - m.PowerGenerated[g,t] <= m.NominalRampDownLimit[g] * m.UnitOn[g,t]  + m.ShutdownRampLimit[g] * m.UnitStop[g,t])

# ======================================= Minimum UpTime & DownTime Constraints ======================================= #
   
    m.MinUpTime_constraints    = ConstraintList(doc='MinUpTime')
    m.MinDownTime_constraints  = ConstraintList(doc='MinDownTime')
    
    for g in m.ThermalGenerators:
        lg = min(W, int(m.InitialTimePeriodsOnline[g]))
        fg = min(W, int(m.InitialTimePeriodsOffline[g]))

        if m.InitialTimePeriodsOnline[g] > 0:
            for t in range(m.InitialTime, min(m.InitialTime + lg, t_fix1+1)):
                m.UnitOn[g, t].fix(1)

        if m.InitialTimePeriodsOffline[g] > 0:
            for t in range(m.InitialTime, min(m.InitialTime + fg, t_fix1+1)):
                m.UnitOn[g, t].fix(0)

        # Intra-window Uptime — only up to the last fixed period
        for t in range(m.InitialTime + lg, min(m.InitialTime + W, t_fix1+1)):
            kg = min(m.InitialTime - t + W, int(m.MinUpTime[g]))
            valid_tt = [tt for tt in range(t, t+kg) if tt in m.TimePeriods and tt <= t_fix1]
            if valid_tt:
                m.MinUpTime_constraints.add(sum(m.UnitOn[g,tt] for tt in valid_tt) >= kg * m.UnitStart[g,t])

        # Intra-window Downtime — only up to last fixed period
        for t in range(m.InitialTime + fg, min(m.FinalTime + 1, t_fix1+1)):
            hg = min(m.FinalTime - t + 1, int(m.MinDownTime[g]))
            valid_tt = [tt for tt in range(t, t+hg) if tt in m.TimePeriods and tt <= t_fix1]
            if valid_tt:
                m.MinDownTime_constraints.add(sum((1 - m.UnitOn[g,tt]) for tt in valid_tt) >= m.UnitStop[g,t] * hg)

# ======================================= Objective Function ======================================= #

    # Cost constraint
    # m.PiecewiseCost = Constraint(m.ThermalGenerators, m.TimePeriods, m.CostSegments, rule=lambda m, g, t, s:
    #     m.PowerCostVar[g, t] >= (m.PowerGenerated[g, t] * data['slp'][g][s-1]) + (m.UnitOn[g, t] * data['intc'][g][s-1]) )

    # m.soc_under = Var(m.StorageUnits, within=NonNegativeReals)
    # for b in m.StorageUnits:
    #     m.Storage_constraints.add(m.soc_under[b] >= m.SoCAtT0[b] - m.SoC[b, t_fix1])

    m.TimePrice = Param(m.TimePeriods, initialize=lambda m, t: 20.0 if (int(t) % 24) in (16, 17, 18, 19, 20) else 5.0)

    def ofv(m):
        start_cost = sum( m.StartUpCost[g] * m.UnitStart[g,t]        for g in m.ThermalGenerators   for t in m.TimePeriods)
        on_cost    = sum( m.CommitmentCost[g] * m.UnitOn[g,t]        for g in m.ThermalGenerators   for t in m.TimePeriods)
        renew_cost = sum( 0.01 * m.RenPowerGenerated[g,t]            for g in m.RenewableGenerators for t in m.TimePeriods)
        power_cost = sum( m.TimePrice[t]  * m.PowerGenerated[g,t]    for g in m.ThermalGenerators   for t in m.TimePeriods)
        shed_cost  = sum( 1000 * m.LoadShed[n,t]                     for n in data["load_buses"]    for t in m.TimePeriods)
        
        #stop_cost  = sum(   m.ShutDownCost[g] * m.UnitStop[g,t]   for g in m.ThermalGenerators for t in m.TimePeriods)
        #c = sum(m.PowerCostVar[g,t] for g in m.ThermalGenerators for t in m.TimePeriods)
        
        return start_cost + on_cost + power_cost + shed_cost + renew_cost # + 5000 * sum(m.soc_under[b] for b in m.StorageUnits) 


        
    m.Objective = Objective(rule=ofv, sense=minimize)

# ======================================= Warm Start ======================================= #

    if warm_start: 
        # for (g,t), v in warm_start['UnitOn'].items():
        #     m.UnitOn[g,t].value = int(round(v))
            
        # for (g,t), v in warm_start['UnitStart'].items():
        #     m.UnitStart[g,t].set_value(int(round(v)))

        # for (g,t), v in warm_start['UnitStop'].items():
        #     m.UnitStop[g,t].set_value(int(round(v)))
        
        for (g,t), v in warm_start['PowerGenerated'].items():
            m.PowerGenerated[g,t].set_value(float(v))
            
        for (g,t), v in warm_start['Flow'].items():
            m.Flow[g,t].set_value(float(v))

        for (g,t), v in warm_start['V_Angle'].items():
            m.V_Angle[g,t].set_value(float(v))

# ======================================= Solve ======================================= #

    # build_time = perf_counter() - t0 # --------- stop BUILD timer 
    # print("build time:", build_time)

    #Solve 
    opt = SolverFactory('gurobi')
    opt.options['MIPGap']     = 0.3
    #opt.options['MIPFocus']   = 2

    # t_attach = perf_counter()       # --------- start ATTACH timer
    # opt.set_instance(m)
    # attach_time = perf_counter() - t_attach
    # print("attach time:", attach_time)

    #t_solve = perf_counter()         # --------- start SOLVE timer
    results = opt.solve(m, tee = False)
    #solve_time = perf_counter() - t_solve
    #print("solve time:", solve_time)

    # t_load = perf_counter()
    # opt.load_vars()
    # load_time = perf_counter() - t_load
    # #print("\nThe objective value is:", round(value(m.Objective), 2))
    # print("load time:", load_time)
    
    t_roll = max(fixed)

    #Get the status at t0 for the next subproblem
    last_status   = { g: int(value(m.UnitOn[g, t_roll]))  for g in m.ThermalGenerators }
    status_change = { g: False for g in m.ThermalGenerators } # Did unit change satus within period
    
    status_dict = {}
    for g in m.ThermalGenerators:
        for t in range(m.InitialTime, t_roll + 1, 1):
            if t == m.InitialTime:
                if value(m.UnitOn[g,t]) - m.UnitOnT0[g] != 0:
                    status_change[g] = True
                    break
            else: 
                if value(m.UnitOn[g,t]) - value(m.UnitOn[g,t-1]) != 0:
                    status_change[g] = True
                    break

        if status_change[g] == False: 
            if m.UnitOnT0[g] == 1:
                status_dict[g] = m.StatusAtT0[g] + ( t_roll - m.InitialTime + 1 )
            else:
                status_dict[g] = m.StatusAtT0[g] - ( t_roll - m.InitialTime + 1 )
        else: 
            streak = 1
            for t in range(t_roll-1, m.InitialTime-1, -1):
                if int(value(m.UnitOn[g,t])) == last_status[g]:
                    streak += 1
                else:
                    break
            if last_status[g] == 1: 
                status_dict[g]  = streak 
            else: 
                status_dict[g] = -streak

    InitialState = {'PowerGeneratedT0':{ g: value(m.PowerGenerated[g, t_roll]) for g in m.ThermalGenerators }, 'StatusAtT0': status_dict, 'SoCT0': { b: value(m.SoC[b, t_roll]) for b in m.StorageUnits } }

    return_object = {'InitialState':InitialState, 
                     'vars':  {  'PowerGenerated': {(g,t): value(m.PowerGenerated[g,t]) for g in m.ThermalGenerators for t in range(m.InitialTime, t_roll+1)}, 
                                         'UnitOn': {(g,t): value(m.UnitOn[g,t])         for g in m.ThermalGenerators for t in range(m.InitialTime, t_roll+1)},
                                      'UnitStart': {(g,t): value(m.UnitStart[g,t])      for g in m.ThermalGenerators for t in range(m.InitialTime, t_roll+1)},
                                       'UnitStop': {(g,t): value(m.UnitStop[g,t])       for g in m.ThermalGenerators for t in range(m.InitialTime, t_roll+1)},
                                     'IsCharging': {(b,t): value(m.IsCharging[b,t])     for b in m.StorageUnits      for t in range(m.InitialTime, t_roll+1)}, 
                                    'ChargePower': {(b,t): value(m.ChargePower[b,t])    for b in m.StorageUnits      for t in range(m.InitialTime, t_roll+1)}, 
                                 'DischargePower': {(b,t): value(m.DischargePower[b,t])    for b in m.StorageUnits      for t in range(m.InitialTime, t_roll+1)}, 
                                            'SoC': {(b,t): value(m.SoC[b,t])            for b in m.StorageUnits      for t in range(m.InitialTime, t_roll+1)}, 
                                  'IsDischarging': {(b,t): value(m.IsDischarging[b,t])  for b in m.StorageUnits      for t in range(m.InitialTime, t_roll+1)}} ,

                        'warm_start': {'PowerGenerated': {(g,t): value(m.PowerGenerated[g,t]) for g in m.ThermalGenerators  for t in range(t_roll+1, m.FinalTime+1)}, 
                                        # 'UnitOn': {(g,t): value(m.UnitOn[g,t])               for g in m.ThermalGenerators for t in range(t_roll+1, m.FinalTime+1)},
                                           'Flow': {(l,t): value(m.Flow[l,t])                 for l in m.TransmissionLines for t in range(t_roll+1, m.FinalTime+1)},
                                        'V_Angle': {(n,t): value(m.V_Angle[n,t])              for n in data['buses']       for t in range(t_roll+1, m.FinalTime+1)}}}
    return return_object
    

