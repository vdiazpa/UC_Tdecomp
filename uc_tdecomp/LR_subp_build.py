
from pyomo.environ import *
import numpy as np

def build_LR_subprobs(data, s_e, index_set):
    
    m = ConcreteModel()

    # Sets
    m.TimePeriods         = s_e
    inner_hrs             = [ x for x in s_e if x != max(s_e) ]
    m.Min_t               = Set(initialize = [min(s_e)-1],       ordered=True)
    m.Max_t               = Set(initialize = [max(s_e)],         ordered=True)
    m.InitialTime         = Set(initialize = [min(s_e)],         ordered=True)
    m.LoadBuses           = Set(initialize = data['load_buses'], ordered=True)
    m.ThermalGenerators   = Set(initialize = data['ther_gens'],  ordered=True)
    m.RenewableGenerators = Set(initialize = data['ren_gens'],   ordered=True)
    m.TransmissionLines   = Set(initialize = data['lines'],      ordered=True)
    m.StorageUnits        = Set(initialize = data['bats'],       ordered=True)

    #m.CostSegments        = Set(initialize=range(1, data['n_segments']), ordered=True)  # number of piecewise cost segments

    m.L_index = Set(initialize = sorted(index_set), ordered=True)   # Lagrange multipliers index set

    # Parameters 
    m.MinUpTime             = Param(m.ThermalGenerators, initialize = data['min_UT'])
    m.MinDownTime           = Param(m.ThermalGenerators, initialize = data['min_DT'])
    m.PowerGeneratedT0      = Param(m.ThermalGenerators, initialize = data['p_init'])   
    m.StatusAtT0            = Param(m.ThermalGenerators, initialize = data['init_status'] ) 
    m.UnitOnT0              = Param(m.ThermalGenerators, initialize = lambda m, g: 1.0 if m.StatusAtT0[g] > 0 else 0.0)
    m.SoCAtT0               = Param(m.StorageUnits,      initialize = data['SoC_init']) 
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

    m.InitialTimePeriodsOnline  = Param(m.ThermalGenerators, 
                                        initialize=lambda m,g: max(0, int(m.MinUpTime[g])   - int(m.StatusAtT0[g])) if m.StatusAtT0[g] > 0 else 0)
    m.InitialTimePeriodsOffline = Param(m.ThermalGenerators, 
                                        initialize=lambda m,g: max(0, int(m.MinDownTime[g]) - abs(int(m.StatusAtT0[g]))) if m.StatusAtT0[g] < 0 else 0)
    
    m.Storage_RoC         = Param(m.StorageUnits, initialize = data['sto_RoC'])
    m.Storage_EnergyCap   = Param(m.StorageUnits, initialize = data['sto_Ecap'])
    m.Storage_Efficiency  = Param(m.StorageUnits, initialize = data['sto_eff'])

    # Variables    
    m.PowerGenerated      = Var(m.ThermalGenerators, m.TimePeriods, within = NonNegativeReals) 
    m.PowerCostVar        = Var(m.ThermalGenerators, m.TimePeriods, within = NonNegativeReals) 
    m.RenPowerGenerated   = Var(m.RenewableGenerators, m.TimePeriods, within=NonNegativeReals) 
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
    
    m.SoC                 = Var(m.StorageUnits,      m.TimePeriods, within = NonNegativeReals, bounds = lambda m, b, t: (0, m.Storage_EnergyCap[b]) )
    m.ChargePower         = Var(m.StorageUnits,      m.TimePeriods, within = NonNegativeReals, bounds = lambda m, b, t: (0, m.Storage_RoC[b] ) )
    m.DischargePower      = Var(m.StorageUnits,      m.TimePeriods, within = NonNegativeReals, bounds = lambda m, b, t: (0, m.Storage_RoC[b] ) )
    m.IsCharging          = Var(m.StorageUnits,      m.TimePeriods, within = Binary)
    m.IsDischarging       = Var(m.StorageUnits,      m.TimePeriods, within = Binary)
    
    # ======================================= Interface Variables and (some) Constraints ======================================= #
    if 1 not in s_e:                                                                                  
        m.PowerGenerated_copy = Var(m.ThermalGenerators, m.Min_t, within = NonNegativeReals) 
        m.LoadShed_copy       = Var(data["load_buses"],  m.Min_t, within = NonNegativeReals)
        m.V_Angle_copy        = Var(     data['buses'],  m.Min_t, within = Reals, bounds = (-180, 180) )
        m.UnitOn_copy         = Var(m.ThermalGenerators, m.Min_t, within = Binary)
        m.Flow_copy           = Var(m.TransmissionLines, m.Min_t, within = Reals,            bounds = lambda m, l, t: (-m.FlowCapacity[l], m.FlowCapacity[l]))
        m.UT_Obl_copy         = Var(m.ThermalGenerators, m.Min_t, within = NonNegativeReals, bounds = lambda m, g, t: (0, m.MinUpTime[g]) )
        m.DT_Obl_copy         = Var(m.ThermalGenerators, m.Min_t, within = NonNegativeReals, bounds = lambda m, g, t: (0, m.MinDownTime[g]) )
        m.SoC_copy            = Var(m.StorageUnits,      m.Min_t, within = NonNegativeReals, bounds = lambda m, b, t: (0, m.Storage_EnergyCap[b]) )

        # Contraints 
        m.MaxCapacity_thermal_copy = Constraint(m.ThermalGenerators, m.Min_t, rule = lambda m, g, t: m.PowerGenerated_copy[g,t] <= m.MaximumPowerOutput[g]*m.UnitOn_copy[g,t], doc= 'max_capacity_thermal_copy')
        m.MinCapacity_thermal_copy = Constraint(m.ThermalGenerators, m.Min_t, rule = lambda m, g, t: m.PowerGenerated_copy[g,t] >= m.MinimumPowerOutput[g]*m.UnitOn_copy[g,t], doc= 'min_capacity_thermal_copy')
        
        #Storage 
        m.SoCbal_constraints_copy  = Constraint(m.StorageUnits, m.Min_t, rule = lambda m, b, t: m.SoC_copy[b,t] <= m.Storage_EnergyCap[b] )

        def nb_copy_rule(m,b,t):
            thermal = sum(m.PowerGenerated_copy[g,t] for g in data["ther_gens_by_bus"][b]) if b in data["ther_gens_by_bus"] else 0.0
            flows   = sum(m.Flow_copy[l,t] * data['lTb'][(l,b)] for l in data['lines_by_bus'][b])
            renew = 0.0 if b not in data['bus_ren_dict'] else sum(m.RenewableOutput[(g,t)] for g in data['bus_ren_dict'][b])
            shed    = m.LoadShed_copy[b,t] if b in data["load_buses"] else 0.0
            return thermal + flows + shed + renew >= data["demand"].get((b,t), 0.0)
        
        m.NodalBalance_copy = Constraint(data["buses"],       m.Min_t, rule = nb_copy_rule)
        m.Power_Flow_copy   = Constraint(m.TransmissionLines, m.Min_t, rule = lambda m, l, t: m.Flow_copy[l,t] * m.LineReactance[l] == m.V_Angle_copy[data["line_ep"][l][0],t] - m.V_Angle_copy[data["line_ep"][l][1],t], doc='Power_flow')

    # ======================================= MUT/MDT Constraints ======================================= #

    m.PerHr_UObl_recurrent = ConstraintList()
    m.PerHr_DObl_recurrent = ConstraintList()

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
            m.UT_Obl_StUB_con  = Constraint(m.ThermalGenerators,          rule = lambda m, g:     m.UT_Obl[g,t] <=      m.MinUpTime[g] * m.UnitOn[g,t])
            m.DT_Obl_StLB_con  = Constraint(m.ThermalGenerators, m.Min_t, rule = lambda m, g, tt: m.DT_Obl[g,t] >= m.DT_Obl_copy[g,tt] + m.MinDownTime[g]*m.UnitStop[g,t] - (1 - m.UnitOn[g,t]))
            m.DT_Obl_StUB_con  = Constraint(m.ThermalGenerators,          rule = lambda m, g:     m.DT_Obl[g,t] <=    m.MinDownTime[g] * (1 - m.UnitOn[g,t]))

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
                
    # ====================================== Battery Energy Storage Constraints ====================================== #
    
    m.Storage_constraints = ConstraintList(doc = 'SoC_constraints')
    
    for b in m.StorageUnits:
        for t in m.TimePeriods:
            m.Storage_constraints.add( m.ChargePower[b,t] <= m.Storage_RoC[b] * m.IsCharging[b,t] )
            m.Storage_constraints.add( m.DischargePower[b,t] <= m.Storage_RoC[b] * m.IsDischarging[b,t] )
            m.Storage_constraints.add( m.IsCharging[b,t] + m.IsDischarging[b,t] <= 1 )

            if t not in m.InitialTime: 
                m.Storage_constraints.add( m.SoC[b,t] == m.SoC[b,t-1] + m.ChargePower[b,t] * m.Storage_Efficiency[b] - m.DischargePower[b,t] / m.Storage_Efficiency[b] )  # SoC balance
                
            else:
                if t == 1: 
                    m.Storage_constraints.add( m.SoC[b,t] == m.SoCAtT0[b] + m.ChargePower[b,t] * m.Storage_Efficiency[b] - m.DischargePower[b,t] / m.Storage_Efficiency[b] )
                else:
                    m.Storage_constraints.add( m.SoC[b,t] == m.SoC_copy[b,t-1] + m.ChargePower[b,t] * m.Storage_Efficiency[b] - m.DischargePower[b,t] / m.Storage_Efficiency[b] )  # SoC balance

    # ======================================= Single Period Constraints ======================================= #
    
    def nb_rule(m,b,t):
        thermal = sum(m.PowerGenerated[g,t] for g in data["ther_gens_by_bus"][b]) if b in data["ther_gens_by_bus"] else 0.0
        flows   = sum(m.Flow[l,t] * data['lTb'][(l,b)] for l in data['lines_by_bus'][b])
        shed    = m.LoadShed[b,t] if b in data["load_buses"] else 0.0
        storage = 0.0 if b not in data['bus_bat'] else sum(m.DischargePower[bat,t] - m.ChargePower[bat,t] for bat in data['bus_bat'][b])
        renew   = 0.0 if b not in data['bus_ren_dict'] else sum(m.RenPowerGenerated[g,t] for g in data['bus_ren_dict'][b])
        return thermal + flows +  shed  + renew + storage >= data["demand"].get((b,t), 0.0)
    
    m.NodalBalance = Constraint(data["buses"], m.TimePeriods, rule = nb_rule)

    for t in m.TimePeriods:
        m.V_Angle[data["ref_bus"], t].fix(0.0)

    m.MaxCapacity_renew = Constraint(m.RenewableGenerators, m.TimePeriods, rule=lambda m,g,t: m.RenPowerGenerated[g,t] <= m.RenewableOutput[(g,t)], doc= 'renewable_output')
    m.MaxCap_thermal    = Constraint(m.ThermalGenerators,   m.TimePeriods, rule=lambda m,g,t: m.PowerGenerated[g,t] <= m.MaximumPowerOutput[g]*m.UnitOn[g,t], doc= 'max_capacity_thermal')
    m.MinCap_thermal    = Constraint(m.ThermalGenerators,   m.TimePeriods, rule=lambda m,g,t: m.PowerGenerated[g,t] >= m.MinimumPowerOutput[g]*m.UnitOn[g,t], doc= 'min_capacity_thermal')
    m.Power_Flow        = Constraint(m.TransmissionLines,   m.TimePeriods, rule=lambda m,l,t: m.Flow[l,t] * m.LineReactance[l] == m.V_Angle[data["line_ep"][l][0],t] - m.V_Angle[data["line_ep"][l][1],t], doc='Power_flow')

    # ======================================= Logic and Ramping ======================================= #
    
    m.logical_constraints  = ConstraintList(doc = 'logical')
    m.RampUp_constraints   = ConstraintList(doc = 'ramp_up')
    m.RampDown_constraints = ConstraintList(doc = 'ramp_down')
    
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
    
     # ======================================= Objective Function with penalties ======================================= #

    if 1 not in s_e:
        timeprice_index = tuple(sorted(set(s_e + [min(s_e)-1])))
    else:
        timeprice_index = tuple(sorted(s_e))

    m.TimePrice = Param(timeprice_index, initialize=lambda m, t: 20.0 if (int(t) % 24) in (16,17,18,19,20) else 5.0)
    
    def ofv_start(m):
        shed_cost  = 1000*sum(        m.LoadShed[n,t]                         for n in data["load_buses"]   for t in m.TimePeriods)
        start_cost = sum(            m.StartUpCost[g] * m.UnitStart[g,t]      for g in m.ThermalGenerators  for t in m.TimePeriods)
        on_cost    = sum(     0.5*m.CommitmentCost[g] * m.UnitOn[g,t]         for g in m.ThermalGenerators  for t in m.Max_t) + sum(m.CommitmentCost[g] * m.UnitOn[g,t]    for g in m.ThermalGenerators   for t in inner_hrs)
        pow_cost   = sum(              m.TimePrice[t] * m.PowerGenerated[g,t] for g in m.ThermalGenerators  for t in m.Max_t) + sum( m.TimePrice[t]  * m.PowerGenerated[g,t]  for g in m.ThermalGenerators   for t in inner_hrs)
        pnlty_on   = sum(       m.L[(g, t, 'UnitOn')] * m.UnitOn[g,t]         for g in m.ThermalGenerators  for t in m.Max_t)   
        pnlty_UObl = sum(       m.L[(g, t, 'UT_Obl')] * m.UT_Obl_end[g,t]     for g in m.ThermalGenerators  for t in m.Max_t)
        pnlty_DObl = sum(       m.L[(g, t, 'DT_Obl')] * m.DT_Obl_end[g,t]     for g in m.ThermalGenerators  for t in m.Max_t)
        pnlty_gen  = sum( m.L[(g,t,'PowerGenerated')] * m.PowerGenerated[g,t] for g in m.ThermalGenerators  for t in m.Max_t) 
        pnlty_soc  = sum(            m.L[(g,t,'SoC')] * m.SoC[g,t]            for g in m.StorageUnits       for t in m.Max_t) 

        return  start_cost  + on_cost + pow_cost + shed_cost +  pnlty_on + pnlty_gen + pnlty_UObl + pnlty_DObl + pnlty_soc

    def ofv(m):
        shed_cost      = 1000*sum(        m.LoadShed[n,t]                              for n in data["load_buses"]   for t in m.TimePeriods)
        st_cost        =      sum(       m.StartUpCost[g] * m.UnitStart[g,t]           for g in m.ThermalGenerators  for t in m.TimePeriods)
        on_cost        = 0.50*sum(    m.CommitmentCost[g] * m.UnitOn[g,t]              for g in m.ThermalGenerators  for t in m.Max_t) + sum( m.CommitmentCost[g] * m.UnitOn[g,t]    for g in m.ThermalGenerators for t in inner_hrs )
        pow_cost       = 0.50*sum(         m.TimePrice[t] * m.PowerGenerated[g,t]      for g in m.ThermalGenerators  for t in m.Max_t) + sum( m.TimePrice[t] * m.PowerGenerated[g,t] for g in m.ThermalGenerators for t in inner_hrs)
        on_cpy_cost    = 0.50*sum(    m.CommitmentCost[g] * m.UnitOn_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t) 
        pow_cpy_cost   = 0.50*sum(         m.TimePrice[t] * m.PowerGenerated_copy[g,t] for g in m.ThermalGenerators  for t in m.Min_t) 
        pnlty_pow      = sum( m.L[(g,t,'PowerGenerated')] * m.PowerGenerated[g,t]      for g in m.ThermalGenerators  for t in m.Max_t) 
        pnlty_on       = sum(       m.L[(g, t, 'UnitOn')] * m.UnitOn[g,t]              for g in m.ThermalGenerators  for t in m.Max_t) 
        pnlty_UObl     = sum(       m.L[(g, t, 'UT_Obl')] * m.UT_Obl_end[g,t]          for g in m.ThermalGenerators  for t in m.Max_t)
        pnlty_DObl     = sum(       m.L[(g, t, 'DT_Obl')] * m.DT_Obl_end[g,t]          for g in m.ThermalGenerators  for t in m.Max_t) 
        pnlty_soc      = sum(          m.L[(g, t, 'SoC')] * m.SoC[g,t]                 for g in m.StorageUnits       for t in m.Max_t)
        pnlty_soc_cpy  = sum(          m.L[(g, t, 'SoC')] * m.SoC_copy[g,t]            for g in m.StorageUnits       for t in m.Min_t)
        pnlty_on_cpy   = sum(       m.L[(g, t, 'UnitOn')] * m.UnitOn_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t)
        pnlty_UObl_cpy = sum(       m.L[(g, t, 'UT_Obl')] * m.UT_Obl_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t)
        pnlty_DObl_cpy = sum(       m.L[(g, t, 'DT_Obl')] * m.DT_Obl_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t)
        pnlty_pow_cpy  = sum( m.L[(g,t,'PowerGenerated')] * m.PowerGenerated_copy[g,t] for g in m.ThermalGenerators  for t in m.Min_t)

        penalty_costs  = pnlty_on + pnlty_pow + pnlty_UObl + pnlty_DObl + pnlty_soc - pnlty_on_cpy - pnlty_pow_cpy - pnlty_UObl_cpy - pnlty_DObl_cpy - pnlty_soc_cpy

        return st_cost + shed_cost + on_cost + on_cpy_cost + pow_cost + pow_cpy_cost + penalty_costs 
    
    def ofv_end(m):
        st_cost        =      sum(      m.StartUpCost[g] * m.UnitStart[g,t]           for g in m.ThermalGenerators  for t in m.TimePeriods)
        shed_cost      = 1000*sum(      m.LoadShed[n, t]                              for n in data["load_buses"]   for t in m.TimePeriods)
        on_cost        = 0.50*sum(   m.CommitmentCost[g] * m.UnitOn_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t) + sum( m.CommitmentCost[g] * m.UnitOn[g,t]       for g in m.ThermalGenerators   for t in m.TimePeriods)
        pow_cost       = 0.50*sum(        m.TimePrice[t] * m.PowerGenerated_copy[g,t] for g in m.ThermalGenerators  for t in m.Min_t) + sum( m.TimePrice[t]  * m.PowerGenerated[g,t]   for g in m.ThermalGenerators   for t in m.TimePeriods)
        pnlty_on_cpy   = sum(      m.L[(g, t, 'UnitOn')] * m.UnitOn_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t)
        pnlty_UObl_cpy = sum(      m.L[(g, t, 'UT_Obl')] * m.UT_Obl_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t)
        pnlty_DObl_cpy = sum(      m.L[(g, t, 'DT_Obl')] * m.DT_Obl_copy[g,t]         for g in m.ThermalGenerators  for t in m.Min_t)
        pnlty_pow_cpy  = sum(m.L[(g,t,'PowerGenerated')] * m.PowerGenerated_copy[g,t] for g in m.ThermalGenerators  for t in m.Min_t)
        pnlty_soc_cpy  = sum(         m.L[(g, t, 'SoC')] * m.SoC_copy[g,t]            for g in m.StorageUnits       for t in m.Min_t)


        return   st_cost + on_cost + pow_cost + shed_cost - 1 * (pnlty_on_cpy + pnlty_pow_cpy + pnlty_UObl_cpy + pnlty_DObl_cpy + pnlty_soc_cpy)
        
    if 1 in s_e: 
        m.Objective = Objective(rule=ofv_start, sense=minimize)
    elif max(s_e) == len(data['periods']):    
        m.Objective = Objective(rule=ofv_end, sense=minimize)
    else: 
        m.Objective = Objective(rule=ofv, sense=minimize)

    return m


