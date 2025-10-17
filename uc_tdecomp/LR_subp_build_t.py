
from pyomo.environ import *
import numpy as np

def build_subprobs_t(data, s_e, index_set):
    
    m = ConcreteModel()

    # Sets
    m.TimePeriods         = s_e
    inner_hrs             = [ x for x in s_e if x != max(s_e) ]
    m.Min_t               = Set(initialize = [min(s_e)-1],     ordered=True)
    m.Max_t               = Set(initialize = [max(s_e)],         ordered=True)
    m.InitialTime         = Set(initialize = [min(s_e)],         ordered=True)
    m.LoadBuses           = Set(initialize = data['load_buses'], ordered=True)
    m.ThermalGenerators   = Set(initialize = data['ther_gens'],  ordered=True)
    m.RenewableGenerators = Set(initialize = data['ren_gens'],   ordered=True)
    m.TransmissionLines   = Set(initialize = data['lines'],      ordered=True)
    #m.CostSegments        = Set(initialize=range(1, data['n_segments']), ordered=True)  # number of piecewise cost segments

    m.L_index = Set(initialize = sorted(index_set), ordered=True)   # Lagrange multipliers index set

    # Parameters 
    m.MinUpTime             = Param(m.ThermalGenerators, initialize = data['min_UT'])
    m.MinDownTime           = Param(m.ThermalGenerators, initialize = data['min_DT'])
    m.PowerGeneratedT0      = Param(m.ThermalGenerators, initialize = data['p_init'])   
    m.StatusAtT0            = Param(m.ThermalGenerators, initialize = data['init_status'] ) 
    m.UnitOnT0              = Param(m.ThermalGenerators, initialize = lambda m, g: 1.0 if m.StatusAtT0[g] > 0 else 0.0)
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
    m.L                     = Param(m.L_index, mutable = True, initialize = 0.0)    # Make dual multipluers a parameter to avoid rebuild. 

    m.InitialTimePeriodsOnline = Param(m.ThermalGenerators, 
                                        initialize = lambda m, g: max(0, int(m.MinUpTime[g]) - int(m.StatusAtT0[g])) if m.StatusAtT0[g] > 0 else 0)
    m.InitialTimePeriodsOffline =  Param(m.ThermalGenerators, 
                                        initialize = lambda m, g: max(0, int(m.MinDownTime[g]) - abs(int(value(m.StatusAtT0[g])))) if m.StatusAtT0[g] < 0 else 0)
    # Variables
    m.PowerGenerated      = Var(m.ThermalGenerators, m.TimePeriods, within = NonNegativeReals) 
    m.PowerCostVar        = Var(m.ThermalGenerators, m.TimePeriods, within = NonNegativeReals) 
    m.UnitOn              = Var(m.ThermalGenerators, m.TimePeriods, within = Binary)
    m.UnitStart           = Var(m.ThermalGenerators, m.TimePeriods, within = Binary)
    m.UnitStop            = Var(m.ThermalGenerators, m.TimePeriods, within = Binary)
    m.V_Angle             = Var(      data['buses'], m.TimePeriods, within = Reals, bounds = (-180, 180) )
    m.Flow                = Var(m.TransmissionLines, m.TimePeriods, within = Reals, bounds = lambda m, l, t: (-m.FlowCapacity[l], m.FlowCapacity[l]))
    m.LoadShed            = Var(data["load_buses"],  m.TimePeriods, within = NonNegativeReals)
    m.UT_Obl              = Var(m.ThermalGenerators, m.TimePeriods, within = NonNegativeReals)
    m.DT_Obl              = Var(m.ThermalGenerators, m.TimePeriods, within = NonNegativeReals)
    m.UT_Obl_end          = Var(m.ThermalGenerators, m.Max_t,       within = NonNegativeReals)
    m.DT_Obl_end          = Var(m.ThermalGenerators, m.Max_t,       within = NonNegativeReals)

    if 1 not in s_e:                                                                                   # Boundary Variable copies 
        m.PowerGenerated_copy = Var(m.ThermalGenerators, m.Min_t, within = NonNegativeReals) 
        m.LoadShed_copy       = Var(data["load_buses"],  m.Min_t, within = NonNegativeReals)
        m.V_Angle_copy        = Var(     data['buses'],  m.Min_t, within = Reals, bounds = (-180, 180) )
        m.UnitOn_copy         = Var(m.ThermalGenerators, m.Min_t, within = Binary)
        m.Flow_copy           = Var(m.TransmissionLines, m.Min_t, within = Reals, bounds = lambda m, l, t: (-m.FlowCapacity[l], m.FlowCapacity[l]))
        m.UT_Obl_copy         = Var(m.ThermalGenerators, m.Min_t, within = NonNegativeReals, bounds = lambda m, g, t: (0, m.MinUpTime[g]) )
        m.DT_Obl_copy         = Var(m.ThermalGenerators, m.Min_t, within = NonNegativeReals, bounds = lambda m, g, t: (0, m.MinDownTime[g]) )

        # Contraints for copy variables
        m.MaxCapacity_thermal_copy = Constraint(m.ThermalGenerators, m.Min_t, rule = lambda m, g, t: m.PowerGenerated_copy[g,t] <= m.MaximumPowerOutput[g]*m.UnitOn_copy[g,t], doc= 'max_capacity_thermal_copy')
        m.MinCapacity_thermal_copy = Constraint(m.ThermalGenerators, m.Min_t, rule = lambda m, g, t: m.PowerGenerated_copy[g,t] >= m.MinimumPowerOutput[g]*m.UnitOn_copy[g,t], doc= 'min_capacity_thermal_copy')

        def nb_copy_rule(m,b,t):
            thermal = sum(m.PowerGenerated_copy[g,t] for g in data["ther_gens_by_bus"][b]) if b in data["ther_gens_by_bus"] else 0.0
            flows   = sum(m.Flow_copy[l,t] * data['lTb'][(l,b)] for l in data['lines_by_bus'][b])
            renew   = data['ren_bus_t'][(b,t)]
            shed    = m.LoadShed_copy[b,t] if b in data["load_buses"] else 0.0
            return thermal + flows + renew + shed >= data["demand"].get((b,t), 0.0)
        
        m.NodalBalance_copy = Constraint(      data["buses"], m.Min_t, rule = nb_copy_rule)
        m.Power_Flow_copy   = Constraint(m.TransmissionLines, m.Min_t, rule = lambda m, l, t: m.Flow_copy[l,t] * m.LineReactance[l] == m.V_Angle_copy[data["line_ep"][l][0],t] - m.V_Angle_copy[data["line_ep"][l][1],t], doc='Power_flow')

    m.UT_Obl_end_con = Constraint(m.ThermalGenerators, m.Max_t, rule = lambda m, g, t: m.UT_Obl_end[g,t] == m.UT_Obl[g,t])
    m.DT_Obl_end_con = Constraint(m.ThermalGenerators, m.Max_t, rule = lambda m, g, t: m.DT_Obl_end[g,t] == m.DT_Obl[g,t])

    for t in m.InitialTime:
        if t == 1: 
            m.UT_Obl_StLB_con  = Constraint(m.ThermalGenerators,  rule = lambda m, g: m.UT_Obl[g,t] >= m.InitialTimePeriodsOnline[g] + m.MinUpTime[g]*m.UnitStart[g,t] - m.UnitOn[g,t])
            m.UT_Obl_StUB_con  = Constraint(m.ThermalGenerators,  rule = lambda m, g: m.UT_Obl[g,t] <= m.MinUpTime[g] * m.UnitOn[g,t])
            m.DT_Obl_StLB_con  = Constraint(m.ThermalGenerators,  rule = lambda m, g: m.DT_Obl[g,t] >= m.InitialTimePeriodsOffline[g] + m.MinDownTime[g]*m.UnitStop[g,t] - (1 - m.UnitOn[g,t]))
            m.DT_Obl_StUB_con  = Constraint(m.ThermalGenerators,  rule = lambda m, g: m.DT_Obl[g,t] <= m.MinDownTime[g] * (1 - m.UnitOn[g,t]))

        else: 
            m.UT_Obl_StLB_con  = Constraint(m.ThermalGenerators, m.Min_t, rule = lambda m, g, tt: m.UT_Obl[g,t] >= m.UT_Obl_copy[g,tt] + m.MinUpTime[g]*m.UnitStart[g,t] - m.UnitOn[g,t])
            m.UT_Obl_StUB_con  = Constraint(m.ThermalGenerators,          rule = lambda m, g:     m.UT_Obl[g,t] <= m.MinUpTime[g]*m.UnitOn[g,t])
            m.DT_Obl_StLB_con  = Constraint(m.ThermalGenerators, m.Min_t, rule = lambda m, g, tt: m.DT_Obl[g,t] >= m.DT_Obl_copy[g,tt] + m.MinDownTime[g]*m.UnitStop[g,t]  - (1 - m.UnitOn[g,t]))
            m.DT_Obl_StUB_con  = Constraint(m.ThermalGenerators,          rule = lambda m, g:     m.DT_Obl[g,t] <= m.MinDownTime[g]*(1 - m.UnitOn[g,t]))

    def nb_rule(m,b,t):
        thermal = sum(m.PowerGenerated[g,t] for g in data["ther_gens_by_bus"][b]) if b in data["ther_gens_by_bus"] else 0.0
        flows   = sum(m.Flow[l,t] * data['lTb'][(l,b)] for l in data['lines_by_bus'][b])
        renew   = data['ren_bus_t'][(b,t)]
        shed    = m.LoadShed[b,t] if b in data["load_buses"] else 0.0
        return thermal + flows + renew + shed >= data["demand"].get((b,t), 0.0)
    
    m.NodalBalance = Constraint(data["buses"], m.TimePeriods, rule = nb_rule)

    for t in m.TimePeriods:
        m.V_Angle[data["ref_bus"], t].fix(0.0)

    # Intra subhorizon constraints
    m.MaxCap_thermal = Constraint(m.ThermalGenerators, m.TimePeriods, rule=lambda m,g,t: m.PowerGenerated[g,t] <= m.MaximumPowerOutput[g]*m.UnitOn[g,t], doc= 'max_capacity_thermal')
    m.MinCap_thermal = Constraint(m.ThermalGenerators, m.TimePeriods, rule=lambda m,g,t: m.PowerGenerated[g,t] >= m.MinimumPowerOutput[g]*m.UnitOn[g,t], doc= 'min_capacity_thermal')
    m.Power_Flow     = Constraint(m.TransmissionLines, m.TimePeriods, 
                                            rule=lambda m,l,t: m.Flow[l,t] * m.LineReactance[l] == m.V_Angle[data["line_ep"][l][0],t] - m.V_Angle[data["line_ep"][l][1],t], doc='Power_flow')

    m.logical_constraints  = ConstraintList(doc = 'logical')
    m.RampUp_constraints   = ConstraintList(doc = 'ramp_up')
    m.RampDown_constraints = ConstraintList(doc = 'ramp_down')
    m.PerHr_UObl_recurrent = ConstraintList(doc = 'UT_hourly_obligations')
    m.PerHr_DObl_recurrent = ConstraintList(doc = 'DT_hourly_obligations')

    for g in m.ThermalGenerators: 
        for t in m.TimePeriods:
            if t == min(s_e): 
                m.PerHr_UObl_recurrent.add( m.UT_Obl[g,t] <= m.MinUpTime[g] * m.UnitOn[g,t] )
                m.PerHr_DObl_recurrent.add( m.DT_Obl[g,t] <= m.MinDownTime[g] * (1 - m.UnitOn[g,t]) )

            else: 
                m.PerHr_UObl_recurrent.add( m.UT_Obl[g,t] >= m.UT_Obl[g,t-1] + m.MinUpTime[g] * m.UnitStart[g,t] - m.UnitOn[g,t] )
                m.PerHr_UObl_recurrent.add( m.UT_Obl[g,t] <= m.MinUpTime[g] * m.UnitOn[g,t] )
                m.PerHr_DObl_recurrent.add( m.DT_Obl[g,t] >= m.DT_Obl[g,t-1] + m.MinDownTime[g] * m.UnitStop[g,t] - (1 - m.UnitOn[g,t]) )
                m.PerHr_DObl_recurrent.add( m.DT_Obl[g,t] <= m.MinDownTime[g] * (1 - m.UnitOn[g,t]) )

    # Add time-coupling constraints
    for g in m.ThermalGenerators:
        for t in m.TimePeriods: 
            if t == 1:
                m.logical_constraints.add(m.UnitStart[g,t] - m.UnitStop[g,t] == m.UnitOn[g,t] - m.UnitOnT0[g])
                m.RampUp_constraints.add(m.PowerGenerated[g,t] - m.PowerGeneratedT0[g]<= m.NominalRampUpLimit[g] * m.UnitOnT0[g] + m.StartupRampLimit[g] * m.UnitStart[g,t])
                m.RampDown_constraints.add(m.PowerGeneratedT0[g] - m.PowerGenerated[g,t] <= m.NominalRampDownLimit[g] * m.UnitOn[g,t] + m.ShutdownRampLimit[g] * m.UnitStop[g,t]) #assumes power generated at 0 is 0

            elif t>1:
                if t != min(m.TimePeriods):
                    m.logical_constraints.add( m.UnitStart[g,t] - m.UnitStop[g,t] == m.UnitOn[g,t] - m.UnitOn[g,t-1])
                    m.RampUp_constraints.add(  m.PowerGenerated[g,t]   - m.PowerGenerated[g,t-1] <= m.NominalRampUpLimit[g]   * m.UnitOn[g,t-1] + m.StartupRampLimit[g]  * m.UnitStart[g,t])
                    m.RampDown_constraints.add( m.PowerGenerated[g,t-1] - m.PowerGenerated[g,t]   <= m.NominalRampDownLimit[g] * m.UnitOn[g,t]   + m.ShutdownRampLimit[g] * m.UnitStop[g,t])
                else:
                    m.logical_constraints.add(m.UnitStart[g,t] - m.UnitStop[g,t] == m.UnitOn[g,t] - m.UnitOn_copy[g,t-1])
                    m.RampUp_constraints.add(m.PowerGenerated[g,t] - m.PowerGenerated_copy[g,t-1] <= m.NominalRampUpLimit[g] * m.UnitOn_copy[g, t-1] + m.StartupRampLimit[g] * m.UnitStart[g,t])
                    m.RampDown_constraints.add(m.PowerGenerated_copy[g,t-1] - m.PowerGenerated[g,t] <= m.NominalRampDownLimit[g] * m.UnitOn[g, t] + m.ShutdownRampLimit[g] * m.UnitStop[g,t])

    def ofv_start(m):
        shed_cost  = 1000*sum(        m.LoadShed[n,t]                         for n in data["load_buses"]   for t in m.TimePeriods)
        start_cost = sum(            m.StartUpCost[g] * m.UnitStart[g,t]      for g in m.ThermalGenerators  for t in m.TimePeriods)
        on_cost    = sum(     0.5*m.CommitmentCost[g] * m.UnitOn[g,t]         for g in m.ThermalGenerators  for t in m.Max_t) + sum(m.CommitmentCost[g] * m.UnitOn[g,t]  for g in m.ThermalGenerators   for t in inner_hrs)
        pow_cost   = 5.00*sum(  m.PowerGenerated[g,t]                         for g in m.ThermalGenerators  for t in m.Max_t) + sum(     10.0  * m.PowerGenerated[g,t]  for g in m.ThermalGenerators   for t in inner_hrs)
        pnlty_on   = sum(       m.L[(g, t, 'UnitOn')] * m.UnitOn[g,t]         for g in m.ThermalGenerators  for t in m.Max_t)   
        pnlty_UObl = sum(       m.L[(g, t, 'UT_Obl')] * m.UT_Obl_end[g,t]     for g in m.ThermalGenerators  for t in m.Max_t)
        pnlty_DObl = sum(       m.L[(g, t, 'DT_Obl')] * m.DT_Obl_end[g,t]     for g in m.ThermalGenerators  for t in m.Max_t)
        pnlty_gen  = sum( m.L[(g,t,'PowerGenerated')] * m.PowerGenerated[g,t] for g in m.ThermalGenerators  for t in m.Max_t) 

        return  start_cost  + on_cost + pow_cost + shed_cost +  pnlty_on + pnlty_gen + pnlty_UObl + pnlty_DObl

    def ofv(m):
        shed_cost      = 1000*sum(        m.LoadShed[n,t]                              for n in data["load_buses"]   for t in m.TimePeriods)
        st_cost        =      sum(       m.StartUpCost[g] * m.UnitStart[g,t]           for g in m.ThermalGenerators  for t in m.TimePeriods)
        on_cost        = 0.50*sum(    m.CommitmentCost[g] * m.UnitOn[g,t]              for g in m.ThermalGenerators  for t in m.Max_t) + sum( m.CommitmentCost[g] * m.UnitOn[g,t] for g in m.ThermalGenerators for t in inner_hrs )
        pow_cost       = 0.50*sum(                   10.0 * m.PowerGenerated[g,t]      for g in m.ThermalGenerators  for t in m.Max_t) + sum( 10 * m.PowerGenerated[g,t] for g in m.ThermalGenerators for t in inner_hrs)
        on_cpy_cost    = 0.50*sum(    m.CommitmentCost[g] * m.UnitOn_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t) 
        pow_cpy_cost   = 0.50*sum(                   10.0 * m.PowerGenerated_copy[g,t] for g in m.ThermalGenerators  for t in m.Min_t) 
        pnlty_pow      = sum( m.L[(g,t,'PowerGenerated')] * m.PowerGenerated[g,t]      for g in m.ThermalGenerators  for t in m.Max_t) 
        pnlty_on       = sum(       m.L[(g, t, 'UnitOn')] * m.UnitOn[g,t]              for g in m.ThermalGenerators  for t in m.Max_t) 
        pnlty_UObl     = sum(       m.L[(g, t, 'UT_Obl')] * m.UT_Obl_end[g,t]          for g in m.ThermalGenerators  for t in m.Max_t)
        pnlty_DObl     = sum(       m.L[(g, t, 'DT_Obl')] * m.DT_Obl_end[g,t]          for g in m.ThermalGenerators  for t in m.Max_t) 
        pnlty_on_cpy   = sum(       m.L[(g, t, 'UnitOn')] * m.UnitOn_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t)
        pnlty_UObl_cpy = sum(       m.L[(g, t, 'UT_Obl')] * m.UT_Obl_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t)
        pnlty_DObl_cpy = sum(       m.L[(g, t, 'DT_Obl')] * m.DT_Obl_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t)
        pnlty_pow_cpy  = sum( m.L[(g,t,'PowerGenerated')] * m.PowerGenerated_copy[g,t] for g in m.ThermalGenerators  for t in m.Min_t)

        penalty_costs  = pnlty_on + pnlty_pow + pnlty_UObl + pnlty_DObl - pnlty_on_cpy - pnlty_pow_cpy - pnlty_UObl_cpy - pnlty_DObl_cpy

        return st_cost + shed_cost + on_cost + on_cpy_cost + pow_cost + pow_cpy_cost + penalty_costs 
    
    def ofv_end(m):
        st_cost        =      sum(      m.StartUpCost[g] * m.UnitStart[g,t]           for g in m.ThermalGenerators  for t in m.TimePeriods)
        shed_cost      = 1000*sum(      m.LoadShed[n, t]                              for n in data["load_buses"]   for t in m.TimePeriods)
        on_cost        = 0.50*sum(   m.CommitmentCost[g] * m.UnitOn_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t) + sum( m.CommitmentCost[g] * m.UnitOn[g,t] for g in m.ThermalGenerators   for t in m.TimePeriods)
        pow_cost       = 0.50*sum(                  10.0 * m.PowerGenerated_copy[g,t] for g in m.ThermalGenerators  for t in m.Min_t) + sum( 10  * m.PowerGenerated[g,t]         for g in m.ThermalGenerators   for t in m.TimePeriods)
        pnlty_on_cpy   = sum(      m.L[(g, t, 'UnitOn')] * m.UnitOn_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t)
        pnlty_UObl_cpy = sum(      m.L[(g, t, 'UT_Obl')] * m.UT_Obl_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t)
        pnlty_DObl_cpy = sum(      m.L[(g, t, 'DT_Obl')] * m.DT_Obl_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t)
        pnlty_pow_cpy  = sum(m.L[(g,t,'PowerGenerated')] * m.PowerGenerated_copy[g,t] for g in m.ThermalGenerators  for t in m.Min_t)

        return   st_cost + on_cost + pow_cost + shed_cost - 1 * (pnlty_on_cpy + pnlty_pow_cpy + pnlty_UObl_cpy + pnlty_DObl_cpy)
        
    if 1 in s_e: 
        m.Objective = Objective(rule=ofv_start, sense=minimize)
    elif max(s_e) == len(data['periods']):    
        m.Objective = Objective(rule=ofv_end, sense=minimize)
    else: 
        m.Objective = Objective(rule=ofv, sense=minimize)

    return m










    # opt = SolverFactory('gurobi')
    # #opt.options['Heuristics'] = 0.5
    # opt.options['Presolve']   = 2
    # opt.options['MIPGap']     = 0.3
    # #opt.options['mipgap'] = 0.1

    # results = opt.solve(m)

    # #print("\nThe objective value is:", round(value(m.Objective), 2))

    # import csv

    # if 1 in s_e: 
    #     return_object = {  'ofv': value(m.Objective), 
    #         'vars': {
    #             'PowerGenerated':  {(g,t): value(m.PowerGenerated[g,t])  for g in m.ThermalGenerators for t in m.TimePeriods },
    #             'UnitOn':          {(g,t): value(m.UnitOn[g,t])          for g in m.ThermalGenerators for t in m.TimePeriods}, 
    #             'UT_Obl_end':      {(g,t): value(m.UT_Obl_end[g,t])      for g in m.ThermalGenerators for t in m.Max_t},
    #             'DT_Obl_end':      {(g,t): value(m.DT_Obl_end[g,t])      for g in m.ThermalGenerators for t in m.Max_t}}}
    # else: 
    #     return_object = {  'ofv': value(m.Objective), 
    #         'vars': {
    #             'UnitOn_copy':         {(g,t): value(m.UnitOn_copy[g,t])         for g in m.ThermalGenerators for t in m.Min_t}, 
    #             'PowerGenerated_copy': {(g,t): value(m.PowerGenerated_copy[g,t]) for g in m.ThermalGenerators for t in m.Min_t},
    #             'UT_Obl_end':          {(g,t): value(m.UT_Obl_end[g,t])          for g in m.ThermalGenerators for t in m.Max_t},
    #             'DT_Obl_end':          {(g,t): value(m.DT_Obl_end[g,t])          for g in m.ThermalGenerators for t in m.Max_t},
    #             'UT_Obl_copy':         {(g,t): value(m.UT_Obl_copy[g,t])         for g in m.ThermalGenerators for t in m.Min_t},
    #             'DT_Obl_copy':         {(g,t): value(m.DT_Obl_copy[g,t])         for g in m.ThermalGenerators for t in m.Min_t},
    #             'PowerGenerated':      {(g,t): value(m.PowerGenerated[g,t])      for g in m.ThermalGenerators for t in m.TimePeriods },
    #             'UnitOn':              {(g,t): value(m.UnitOn[g,t])              for g in m.ThermalGenerators for t in m.TimePeriods }}}
        
    # if k ==1:
    #     m.write(f'm.{str(s_e)}_{k}.lp', io_options = {'symbolic_solver_labels': True})
    #     csv_filename = f"solution_{str(s_e)}_{k}.csv"
    #     with open(csv_filename, mode='w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(['Var', 'g', 't', 'Value'])

    #         for varname, index_dict in return_object['vars'].items():
    #             for (g,t), val in index_dict.items():
    #                 writer.writerow([varname, g, t, val])

    # return return_object
    

