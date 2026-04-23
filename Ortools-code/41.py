import json
from ortools.linear_solver import pywraplp

inp = {
    "initial_capital": 500000,
    "horizon_years": 3,
    "products": [
        {
            "id": "p1",
            "available_at_start_of_years": [1, 2, 3],
            "maturity_end_of_year": 1,
            "payoff_multiplier": 1.2,
            "capacity": None
        },
        {
            "id": "p2",
            "available_at_start_of_years": [1],
            "maturity_end_of_year": 2,
            "payoff_multiplier": 1.5,
            "capacity": 120000
        },
        {
            "id": "p3",
            "available_at_start_of_years": [2],
            "maturity_end_of_year": 2,
            "payoff_multiplier": 1.6,
            "capacity": 150000
        },
        {
            "id": "p4",
            "available_at_start_of_years": [3],
            "maturity_end_of_year": 3,
            "payoff_multiplier": 1.4,
            "capacity": 100000
        }
    ]
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver.")

INF = solver.infinity()

# Decision variables
x11 = solver.NumVar(0.0, INF, "x11")      # p1 at start of Year 1
x12 = solver.NumVar(0.0, INF, "x12")      # p1 at start of Year 2
x13 = solver.NumVar(0.0, INF, "x13")      # p1 at start of Year 3
x21 = solver.NumVar(0.0, 120000.0, "x21") # p2 at start of Year 1
x32 = solver.NumVar(0.0, 150000.0, "x32") # p3 at start of Year 2
x43 = solver.NumVar(0.0, 100000.0, "x43") # p4 at start of Year 3

# Cash carryover (idle cash kept without interest)
s1 = solver.NumVar(0.0, INF, "s1")  # carry from end of Year 1 to start of Year 2
s2 = solver.NumVar(0.0, INF, "s2")  # carry from end of Year 2 to start of Year 3

# Budget / flow constraints
solver.Add(x11 + x21 + s1 == inp["initial_capital"])
solver.Add(x12 + x32 + s2 == 1.2 * x11 + s1)
solver.Add(x13 + x43 == 1.5 * x21 + 1.6 * x32 + s2 + x12 * 1.2)

# Objective: maximize total wealth at end of Year 3
ending_wealth = 1.2 * x13 + 1.4 * x43
solver.Maximize(ending_wealth)

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

def val(v):
    return round(v.solution_value(), 6)

output = {
    "investments": [
        {"product_id": "p1", "start_year": 1, "amount": val(x11)},
        {"product_id": "p2", "start_year": 1, "amount": val(x21)},
        {"product_id": "p1", "start_year": 2, "amount": val(x12)},
        {"product_id": "p3", "start_year": 2, "amount": val(x32)},
        {"product_id": "p1", "start_year": 3, "amount": val(x13)},
        {"product_id": "p4", "start_year": 3, "amount": val(x43)}
    ],
    "cash_carryover": [
        {"from_year_end": 1, "amount": val(s1)},
        {"from_year_end": 2, "amount": val(s2)}
    ],
    "ending_wealth": round(solver.Objective().Value(), 6)
}

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": round(solver.Objective().Value(), 6) if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else None,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))