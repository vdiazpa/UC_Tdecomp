#benders.py
import pandas as pd, sys
sys.path.insert(0, "C:\\Users\\vdiazpa\\mpi-sppy")
from ..opt.bench_UC import benchmark_UC_build
import mpisppy.utils.sputils as sputils
from mpisppy.opt.lshaped import LShapedMethod
from pyomo.environ import *


def scenario_creator(scenario_name, **kwargs):
    
    data = kwargs.get("data")
    fixed_commitment = kwargs.get("fixed_commitment")
    m = benchmark_UC_build(data, fixed_commitment=fixed_commitment, do_solve = False)
    sputils.attach_root_node(m, m.StageOneCost, [m.UnitOn, m.UnitStart, m.UnitStop, m.IsCharging, m.IsDischarging]) # ---- Attach Non-anticipativity info ---
    m._mpisppy_probability = 1.0 

    return m        

def model_build_solve_benders_mpi(data, max_iter = 30, fixed_commitment = None):

    options = {"root_solver" : "gurobi_persistent", "root_tee": True, "root_solver_options": { "MIPGap": 0.01}, # master problem # other options: "OutputFlag":1, "LogToConsole": 1,
               "sp_solver"   : "gurobi",   "sp_tee": True, "store_subproblems" : True,  # subproblems
               "verbose": True,  "display_progress": True, "max_iter": max_iter, "tol": 1e-4, "valid_eta_lb": {"sc1": 0.0 } } #
    
    all_scenario_names = ["sc1"]
    ls = LShapedMethod(options, all_scenario_names, scenario_creator, scenario_creator_kwargs={"data":data, "fixed_commitment": fixed_commitment})
    result = ls.lshaped_algorithm()

    sp = ls.subproblems["sc1"]

    # quick check: are "would-be binaries" fractional in the subproblem?
    def show_frac_var(comp, name, max_show=10, tol=1e-6):
        if not hasattr(sp, name):
            print(f"sp has no {name}")
            return
        v = getattr(sp, name)
        shown = 0
        for idx in v:
            val = v[idx].value
            if val is not None and tol < val < 1 - tol:
                print(f"fractional {name}{idx} = {val}")
                shown += 1
                if shown >= max_show:
                    break
        print(f"{name}: shown {shown}")

    show_frac_var(sp, "UnitStart")
    show_frac_var(sp, "UnitStop")


    print("Number of dual entries:", len(sp.dual))

    shown = 0
    for c in sp.component_data_objects(Constraint, active=True):
        if c in sp.dual:
            val = sp.dual[c]
            if abs(val) > 1e-8:
                shown+=1
                if shown >= 5:
                    break

    if shown==0:
        print("Non nonzero duals found")

def solve_MILP(model, tee=False, opt_gap = 0.01):
    opt = SolverFactory('gurobi')
    opt.options['MIPGap'] = opt_gap
    opt.solve(model, tee=tee)
    print(value(model.Objective))

def build_master(data):

    m = ConcreteModel()

    #Sets
    m.TimePeriods         = data['periods'] 
    m.InitialTime         = min(m.TimePeriods)
    m.FinalTime           = max(m.TimePeriods)
    m.LoadBuses           = Set(initialize=data['load_buses'], ordered=True)
    m.ThermalGenerators   = Set(initialize=data['ther_gens'], ordered=True)

    #Params
    W_full              = m.FinalTime - m.InitialTime + 1
    m.MinUpTime         = Param(m.ThermalGenerators, initialize = data['min_UT'])
    m.MinDownTime       = Param(m.ThermalGenerators, initialize = data['min_DT'])
    m.PowerGeneratedT0  = Param(m.ThermalGenerators, initialize = data['p_init'])   
    m.StatusAtT0        = Param(m.ThermalGenerators, initialize = data['init_status'] ) 
    m.UnitOnT0          = Param(m.ThermalGenerators, initialize = lambda m, g: 1.0 if m.StatusAtT0[g] > 0 else 0.0)
    #m.SoCAtT0           = Param(m.StorageUnits,      initialize = data['SoC_init']) 
    m.CommitmentCost    = Param(m.ThermalGenerators, initialize=data['commit_cost'])
    m.StartUpCost       = Param(m.ThermalGenerators, initialize=data['startup_cost'])

    m.InitialTimePeriodsOnline = Param(m.ThermalGenerators, 
        initialize = lambda m, g: (min(W_full, max(0, int(m.MinUpTime[g]) - int(m.StatusAtT0[g]))) if m.StatusAtT0[g] > 0 else 0))
    
    m.InitialTimePeriodsOffline = Param(m.ThermalGenerators, 
        initialize = lambda m, g: (min(W_full, max(0, int(m.MinDownTime[g]) - abs(int(m.StatusAtT0[g])))) if m.StatusAtT0[g] < 0 else 0))

    #Vars
    m.UTRemain            = Var(m.ThermalGenerators, m.TimePeriods, within=NonNegativeReals) 
    m.DTRemain            = Var(m.ThermalGenerators, m.TimePeriods, within=NonNegativeReals) 
    m.UnitOn              = Var(m.ThermalGenerators, m.TimePeriods, within=Binary)
    m.UnitStart           = Var(m.ThermalGenerators, m.TimePeriods, within=Binary)
    m.UnitStop            = Var(m.ThermalGenerators, m.TimePeriods, within=Binary)
    m.eta = Var(within = NonNegativeReals) # Eta: LB on dispatch cost

    #Constraints

    m.logical = ConstraintList()
    m.UTc = ConstraintList() 
    m.DTc = ConstraintList()

    for g in m.ThermalGenerators:
        for t in m.TimePeriods: 
            if t == m.InitialTime:
                m.logical.add(m.UnitStart[g,t] - m.UnitStop[g,t] == m.UnitOn[g,t] - m.UnitOnT0[g])
                m.UTc.add( m.UTRemain[g,t] >=  m.InitialTimePeriodsOnline[g] +m.MinUpTime[g]*m.UnitStart[g,t] - m.UnitOn[g,t])
                m.DTc.add( m.DTRemain[g,t] >=  m.InitialTimePeriodsOffline[g] +m.MinDownTime[g]*m.UnitStop[g,t] - (1 -m.UnitOn[g,t]))

            else: 
                m.logical.add(m.UnitStart[g,t] - m.UnitStop[g,t] == m.UnitOn[g,t] - m.UnitOn[g,t-1])
                m.UTc.add( m.UTRemain[g,t] >=  m.UTRemain[g,t-1] +m.MinUpTime[g]*m.UnitStart[g,t] - m.UnitOn[g,t])
                m.DTc.add( m.DTRemain[g,t] >=  m.DTRemain[g,t-1] +m.MinDownTime[g]*m.UnitStop[g,t] - (1 -m.UnitOn[g,t]))


        lg = min(m.FinalTime, int(m.InitialTimePeriodsOnline[g]))         # CarryOver Uptime    
        if m.InitialTimePeriodsOnline[g] > 0:
            for t in range(m.InitialTime,  m.InitialTime + lg):
                m.UnitOn[g, t].fix(1)

        fg = min(m.FinalTime, int(m.InitialTimePeriodsOffline[g]))       # CarryOver Downtime
        if m.InitialTimePeriodsOffline[g] > 0: 
            for t in range(m.InitialTime, m.InitialTime + fg):
                m.UnitOn[g, t].fix(0)

    # Objective
    m.StageOneCost = Expression(expr = sum( m.StartUpCost[g] * m.UnitStart[g,t]     for g in m.ThermalGenerators   for t in m.TimePeriods)
                                         + sum( m.CommitmentCost[g] * m.UnitOn[g,t]   for g in m.ThermalGenerators   for t in m.TimePeriods) )
    m.Objective = Objective(expr = m.StageOneCost + m.eta, sense = minimize)

    m.OptCuts = ConstraintList()
    m.FeasCuts = ConstraintList()

    return m

def build_subproblems(data): 

    m = ConcreteModel()
    m.dual = Suffix(direction=Suffix.IMPORT)

    #sets
    m.TimePeriods         = data['periods'] 
    m.LoadBuses           = Set(initialize=data['load_buses'], ordered=True)
    m.InitialTime         = min(m.TimePeriods)
    m.FinalTime           = max(m.TimePeriods)
    m.ThermalGenerators   = Set(initialize=data['ther_gens'], ordered=True)
    m.RenewableGenerators = Set(initialize=data['ren_gens'],  ordered=True)
    m.Generators          = Set(initialize=data['gens'],      ordered=True)
    m.TransmissionLines   = Set(initialize=data['lines'],     ordered=True)
    #m.StorageUnits        = Set(initialize = data['bats'], ordered=True)

    #Params
    W_full                  = m.FinalTime - m.InitialTime + 1
    m.PowerGeneratedT0   = Param(m.ThermalGenerators, initialize = data['p_init'])   
    m.StatusAtT0         = Param(m.ThermalGenerators, initialize = data['init_status'] ) 
    m.UnitOnT0           = Param(m.ThermalGenerators, initialize = lambda m, g: 1.0 if m.StatusAtT0[g] > 0 else 0.0)
    #m.SoCAtT0            = Param(m.StorageUnits,      initialize = data['SoC_init']) 
    m.MaximumPowerOutput    = Param(m.ThermalGenerators,   initialize=data['p_max'])
    m.MinimumPowerOutput    = Param(m.ThermalGenerators,   initialize=data['p_min'])
    m.RenewableOutput       = Param(m.RenewableGenerators, data["periods"], initialize=data['ren_output'])
    m.NominalRampUpLimit    = Param(m.ThermalGenerators,   initialize=data['rup'])
    m.NominalRampDownLimit  = Param(m.ThermalGenerators,   initialize=data['rdn'])
    m.StartupRampLimit      = Param(m.ThermalGenerators,   initialize=data['suR'])
    m.ShutdownRampLimit     = Param(m.ThermalGenerators,   initialize=data['sdR'])
    m.FlowCapacity          = Param(m.TransmissionLines,   initialize = data['line_cap'])
    m.LineReactance         = Param(m.TransmissionLines,   initialize = data['line_reac'])
    # m.Storage_RoC           = Param(m.StorageUnits,        initialize = data['sto_RoC'])
    # m.Storage_EnergyCap     = Param(m.StorageUnits,        initialize = data['sto_Ecap'])
    # m.Storage_Efficiency    = Param(m.StorageUnits,        initialize = data['sto_eff'])

     # =======
    #Copues of master variables 
    m.U  = Var(m.ThermalGenerators, m.TimePeriods, within = Reals)
    m.SU = Var(m.ThermalGenerators, m.TimePeriods, within = Reals)
    m.SD = Var(m.ThermalGenerators, m.TimePeriods, within = Reals)

    m.FixU  = ConstraintList()
    m.FixSU = ConstraintList()
    m.FixSD = ConstraintList()

    #Dispatch vars
    m.PowerGenerated      = Var(m.ThermalGenerators, m.TimePeriods, within=NonNegativeReals) 
    m.RenPowerGenerated   = Var(m.RenewableGenerators, m.TimePeriods, within=NonNegativeReals) 
    m.V_Angle             = Var(data['buses'],       m.TimePeriods, within = Reals, bounds = (-180, 180) )
    m.Flow                = Var(m.TransmissionLines, m.TimePeriods, within = Reals, bounds = lambda m, l, t: (-value(m.FlowCapacity[l]), value(m.FlowCapacity[l])))
    m.LoadShed            = Var(data["load_buses"],  m.TimePeriods, within = NonNegativeReals)

    #Constraints
    m.MaxCapacity_thermal = Constraint(m.ThermalGenerators,   m.TimePeriods, rule=lambda m,g,t: m.PowerGenerated[g,t] <= m.MaximumPowerOutput[g]*m.U[g,t], doc= 'max_capacity_thermal')
    m.MinCapacity_thermal = Constraint(m.ThermalGenerators,   m.TimePeriods, rule=lambda m,g,t: m.PowerGenerated[g,t] >= m.MinimumPowerOutput[g]*m.U[g,t], doc= 'min_capacity_thermal')
    m.MaxCapacity_renew   = Constraint(m.RenewableGenerators, m.TimePeriods, rule=lambda m,g,t: m.RenPowerGenerated[g,t] <= m.RenewableOutput[(g,t)], doc= 'renewable_output')
    m.Power_Flow          = Constraint(m.TransmissionLines,   m.TimePeriods, rule=lambda m,l,t: m.Flow[l,t]*m.LineReactance[l] == m.V_Angle[data["line_ep"][l][0],t] - m.V_Angle[data["line_ep"][l][1],t], doc='Power_flow')

    def nb_rule(m, b, t):
        thermal = sum(m.PowerGenerated[g,t] for g in data["ther_gens_by_bus"].get(b, []))
        flows   = sum(m.Flow[l,t] * data['lTb'][(l,b)] for l in data['lines_by_bus'][b])
        renew   = 0.0 if b not in data['bus_ren_dict'] else sum(m.RenPowerGenerated[g,t] for g in data['bus_ren_dict'][b])
        shed    = m.LoadShed[b,t] if b in data["load_buses"] else 0.0
        #storage = 0.0 if b not in data['bus_bat'] else sum(m.DischargePower[bat,t] - m.ChargePower[bat,t] for bat in data['bus_bat'][b])
        return thermal + flows + renew + shed  == data["demand"].get((b,t), 0.0)
    
    m.NodalBalance = Constraint(data["buses"], m.TimePeriods, rule = nb_rule)

    m.RampUp_constraints   = ConstraintList(doc = 'ramp_up')
    m.RampDown_constraints = ConstraintList(doc = 'ramp_down')

    for g in m.ThermalGenerators:
        for t in m.TimePeriods: 
            if t == m.InitialTime:
                m.RampUp_constraints.add(m.PowerGenerated[g,t] - m.PowerGeneratedT0[g]<= m.NominalRampUpLimit[g] * m.UnitOnT0[g] + m.StartupRampLimit[g] * m.SU[g,t])
                m.RampDown_constraints.add(m.PowerGeneratedT0[g] - m.PowerGenerated[g,t] <= m.NominalRampDownLimit[g] * m.U[g,t] + m.ShutdownRampLimit[g] * m.SD[g,t]) #assumes power generated at 0 is 0

            else: 
                m.RampUp_constraints.add(m.PowerGenerated[g,t] - m.PowerGenerated[g,t-1] <= m.NominalRampUpLimit[g] * m.U[g, t-1] + m.StartupRampLimit[g] * m.SU[g,t])
                m.RampDown_constraints.add(m.PowerGenerated[g,t-1] - m.PowerGenerated[g,t] <= m.NominalRampDownLimit[g] * m.U[g, t] + m.ShutdownRampLimit[g] * m.SD[g,t])
                
    def ofv(m):
        power_cost = sum( data['gen_cost'][g]  * m.PowerGenerated[g,t]     for g in m.ThermalGenerators   for t in m.TimePeriods)
        renew_cost = sum( 0.01 * m.RenPowerGenerated[g,t]                  for g in m.RenewableGenerators for t in m.TimePeriods)
        shed_cost  = sum( 1000 * m.LoadShed[n,t]                           for n in data["load_buses"]    for t in m.TimePeriods)

        return   power_cost + shed_cost + renew_cost 
          
    m.Objective = Objective(rule=ofv, sense=minimize)
    
    return m

def solve_subproblem_lp(sp, master, solver_name = "gurobi"):
    sp.del_component(sp.FixU);  sp.FixU  = ConstraintList()
    sp.del_component(sp.FixSU); sp.FixSU = ConstraintList()
    sp.del_component(sp.FixSD); sp.FixSD = ConstraintList()
        
    for g in sp.ThermalGenerators:
        for t in sp.TimePeriods: 
            UStar  = value(master.UnitOn[g,t])
            SUStar = value(master.UnitStart[g,t])
            SDStar = value(master.UnitStop[g,t])

            sp.FixU.add(sp.U[g,t] == UStar)
            sp.FixSU.add(sp.SU[g,t] == SUStar)
            sp.FixSD.add(sp.SD[g,t] == SDStar)

    opt = SolverFactory(solver_name)
    res = opt.solve(sp, tee=False, suffixes=["dual"])

    nnz = 0
    for con in list(sp.FixU.values()) +  list(sp.FixSU.values()) +  list(sp.FixSD.values()): 
        if con in sp.dual and abs(sp.dual[con]) > 1e-8:
            nnz+=1

    print("Nonzero duals on fix constraints:", nnz)
    return res

def run_benders(data, master_solver = "gurobi", sub_solver = "gurobi", max_iter = 50, tol = 1e-3 ):

    M = build_master(data)
    SP = build_subproblems(data)
    optM = SolverFactory(master_solver)
    
    LB = -1e100
    UB = 1e100

    for k in range(1, max_iter +1):
        # -- solve master
        optM.solve(M, tee=True)
        LB = max(LB, value(M.Objective))

        # -- solve subproblem at current commitment
        res_sp = solve_subproblem_lp(SP, M, solver_name = sub_solver)

        Q  = value(SP.Objective)  # Dispatch cost at current commitment
        UB = min(UB, value(M.StageOneCost) + Q)

        gap = UB - LB

        print(f"\nIter {k}: LB={LB:.4f}  UB={UB:.4f}  gap={gap:.4f}")

        if gap <=tol:
            print("Converged.")
            break

        # -- Build optimality cut: eta >= Q + sum(phi*(x-x*))
 
        expr = Q
        
        duals_fixU  = list(SP.FixU.values())
        duals_fixSU = list(SP.FixSU.values())
        duals_fixSD = list(SP.FixSD.values())

        # Build cut expression

        #U terms
        c_i = 0
        for g in SP.ThermalGenerators: 
            for t in SP.TimePeriods:
                con=duals_fixU[c_i]
                phi = SP.dual[con] if con in SP.dual else 0.0
                Ustar = value(M.UnitOn[g,t])
                expr+= -phi * (M.UnitOn[g,t] - Ustar)
                c_i +=1

        #SU terms
        c_i = 0
        for g in SP.ThermalGenerators: 
            for t in SP.TimePeriods:
                con=duals_fixSU[c_i]
                phi = SP.dual[con] if con in SP.dual else 0.0
                SUstar = value(M.UnitStart[g,t])
                expr+= -phi * (M.UnitStart[g,t] - SUstar)
                c_i +=1    

        #SU terms
        c_i = 0
        for g in SP.ThermalGenerators: 
            for t in SP.TimePeriods:
                con=duals_fixSD[c_i]
                phi = SP.dual[con] if con in SP.dual else 0.0
                SDstar = value(M.UnitStop[g,t])
                expr+= -phi * (M.UnitStop[g,t] - SDstar)
                c_i +=1                

        # Add cut
        M.OptCuts.add(M.eta>=expr)

    return M, SP
