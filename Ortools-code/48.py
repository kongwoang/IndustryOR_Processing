# Source: :contentReference[oaicite:0]{index=0}
import json
from ortools.linear_solver import pywraplp

inp = {
    "products": ["A", "B"],
    "processes": {
        "I": {
            "hours_per_unit": {
                "A": 4,
                "B": 6
            },
            "weekly_capacity": 150
        },
        "II": {
            "hours_per_unit": {
                "A": 3,
                "B": 2
            },
            "weekly_capacity": 70,
            "max_overtime": 30
        }
    },
    "regular_profit_per_unit": {
        "A": 300,
        "B": 450
    },
    "overtime_profit_reduction_per_unit": {
        "A": 20,
        "B": 25
    },
    "goals": {
        "min_weekly_profit": 10000,
        "min_weekly_production": {
            "A": 10,
            "B": 15
        },
        "process_I_target_hours": 150,
        "process_II_regular_target_hours": 70
    },
    "priority_order": [
        "minimize_profit_shortfall",
        "minimize_contract_shortfall",
        "minimize_underutilization_of_process_I_and_regular_process_II",
        "minimize_overtime_profit_reduction"
    ]
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("GLOP solver is not available.")

inf = solver.infinity()
tol = 1e-9

# Production split by whether Process II uses regular time or overtime
ar = solver.NumVar(0.0, inf, "A_regular")
br = solver.NumVar(0.0, inf, "B_regular")
ao = solver.NumVar(0.0, inf, "A_overtime")
bo = solver.NumVar(0.0, inf, "B_overtime")

# Deviational variables for preemptive goal programming
d_profit_shortfall = solver.NumVar(0.0, inf, "d_profit_shortfall")
d_A_shortfall = solver.NumVar(0.0, inf, "d_A_shortfall")
d_B_shortfall = solver.NumVar(0.0, inf, "d_B_shortfall")
d_I_under = solver.NumVar(0.0, inf, "d_I_under")
d_II_regular_under = solver.NumVar(0.0, inf, "d_II_regular_under")

proc_I = inp["processes"]["I"]["hours_per_unit"]
proc_II = inp["processes"]["II"]["hours_per_unit"]
profit = inp["regular_profit_per_unit"]
reduction = inp["overtime_profit_reduction_per_unit"]
goals = inp["goals"]

A_total_expr = ar + ao
B_total_expr = br + bo

gross_profit_expr = profit["A"] * A_total_expr + profit["B"] * B_total_expr
overtime_reduction_expr = reduction["A"] * ao + reduction["B"] * bo
net_profit_expr = gross_profit_expr - overtime_reduction_expr

I_total_expr = proc_I["A"] * A_total_expr + proc_I["B"] * B_total_expr
II_regular_expr = proc_II["A"] * ar + proc_II["B"] * br
II_overtime_expr = proc_II["A"] * ao + proc_II["B"] * bo

# Goal p1: profit >= 10000
solver.Add(net_profit_expr + d_profit_shortfall >= goals["min_weekly_profit"])

# Goal p2: contract minimum production
solver.Add(A_total_expr + d_A_shortfall >= goals["min_weekly_production"]["A"])
solver.Add(B_total_expr + d_B_shortfall >= goals["min_weekly_production"]["B"])

# Goal p3: Process I exactly 150 hours; regular Process II ideally fully utilized at 70 hours
solver.Add(I_total_expr + d_I_under == goals["process_I_target_hours"])
solver.Add(II_regular_expr + d_II_regular_under == goals["process_II_regular_target_hours"])

# Goal p4: overtime in Process II allowed up to 30 hours
solver.Add(II_overtime_expr <= inp["processes"]["II"]["max_overtime"])

objective = solver.Objective()

def solve_minimize(terms):
    objective.Clear()
    for var, coeff in terms:
        objective.SetCoefficient(var, coeff)
    objective.SetMinimization()
    return solver.Solve(), objective.Value()

def status_name(code):
    mapping = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    return mapping.get(code, f"STATUS_{code}")

status, p1 = solve_minimize([(d_profit_shortfall, 1.0)])
if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    raise RuntimeError(status_name(status))
solver.Add(d_profit_shortfall <= p1 + tol)

status, p2 = solve_minimize([(d_A_shortfall, 1.0), (d_B_shortfall, 1.0)])
if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    raise RuntimeError(status_name(status))
solver.Add(d_A_shortfall + d_B_shortfall <= p2 + tol)

status, p3 = solve_minimize([(d_I_under, 1.0), (d_II_regular_under, 1.0)])
if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    raise RuntimeError(status_name(status))
solver.Add(d_I_under + d_II_regular_under <= p3 + tol)

status, p4 = solve_minimize([(ao, reduction["A"]), (bo, reduction["B"])])
if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    raise RuntimeError(status_name(status))

A_regular = ar.solution_value()
B_regular = br.solution_value()
A_overtime = ao.solution_value()
B_overtime = bo.solution_value()

A_total = A_regular + A_overtime
B_total = B_regular + B_overtime

gross_profit = profit["A"] * A_total + profit["B"] * B_total
overtime_reduction = reduction["A"] * A_overtime + reduction["B"] * B_overtime
net_profit = gross_profit - overtime_reduction

I_total = proc_I["A"] * A_total + proc_I["B"] * B_total
II_regular = proc_II["A"] * A_regular + proc_II["B"] * B_regular
II_overtime = proc_II["A"] * A_overtime + proc_II["B"] * B_overtime
II_total = II_regular + II_overtime

def r(x):
    return round(float(x), 6)

output = {
    "regular_production": {
        "A": r(A_regular),
        "B": r(B_regular)
    },
    "overtime_production": {
        "A": r(A_overtime),
        "B": r(B_overtime)
    },
    "total_production": {
        "A": r(A_total),
        "B": r(B_total)
    },
    "profit": {
        "gross_before_overtime_reduction": r(gross_profit),
        "overtime_reduction": r(overtime_reduction),
        "net": r(net_profit)
    },
    "process_hours": {
        "I_total": r(I_total),
        "II_regular": r(II_regular),
        "II_overtime": r(II_overtime),
        "II_total": r(II_total)
    },
    "goal_deviations": {
        "profit_shortfall": r(d_profit_shortfall.solution_value()),
        "A_contract_shortfall": r(d_A_shortfall.solution_value()),
        "B_contract_shortfall": r(d_B_shortfall.solution_value()),
        "process_I_underutilization": r(d_I_under.solution_value()),
        "process_II_regular_underutilization": r(d_II_regular_under.solution_value())
    },
    "priority_objectives": {
        "p1": r(p1),
        "p2": r(p2),
        "p3": r(p3),
        "p4": r(p4)
    }
}

print(json.dumps({
    "status": status_name(status),
    "objective_value": r(net_profit),
    "example_output": output
}, ensure_ascii=False))