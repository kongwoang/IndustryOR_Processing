import json
from ortools.linear_solver import pywraplp

# Source: :contentReference[oaicite:0]{index=0}

inp = {
    "total_acres": 120,
    "profits_per_acre": {
        "apples": 2000,
        "pears": 1800,
        "oranges": 2200,
        "lemons": 3000
    },
    "constraints": {
        "apples_min_at_least_multiplier_of_pears": 2,
        "apples_min_at_least_multiplier_of_lemons": 3,
        "oranges_equals_multiplier_of_lemons": 2,
        "max_fruit_types": 2
    }
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

fruits = ["apples", "pears", "oranges", "lemons"]
M = inp["total_acres"]

x = {f: solver.NumVar(0.0, solver.infinity(), f"x_{f}") for f in fruits}
y = {f: solver.BoolVar(f"y_{f}") for f in fruits}

# Total land
solver.Add(sum(x[f] for f in fruits) <= inp["total_acres"])

# Logical linkage for "no more than two fruit types"
for f in fruits:
    solver.Add(x[f] <= M * y[f])

solver.Add(sum(y[f] for f in fruits) <= inp["constraints"]["max_fruit_types"])

# Problem constraints
solver.Add(
    x["apples"] >=
    inp["constraints"]["apples_min_at_least_multiplier_of_pears"] * x["pears"]
)
solver.Add(
    x["apples"] >=
    inp["constraints"]["apples_min_at_least_multiplier_of_lemons"] * x["lemons"]
)
solver.Add(
    x["oranges"] ==
    inp["constraints"]["oranges_equals_multiplier_of_lemons"] * x["lemons"]
)

# Objective: maximize profit
objective = solver.Objective()
for f in fruits:
    objective.SetCoefficient(x[f], inp["profits_per_acre"][f])
objective.SetMaximization()

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

def clean(v):
    return 0.0 if abs(v) < 1e-9 else float(v)

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "apples_acres": clean(x["apples"].solution_value()),
        "pears_acres": clean(x["pears"].solution_value()),
        "oranges_acres": clean(x["oranges"].solution_value()),
        "lemons_acres": clean(x["lemons"].solution_value())
    }
    objective_value = float(solver.Objective().Value())
else:
    output = {
        "apples_acres": None,
        "pears_acres": None,
        "oranges_acres": None,
        "lemons_acres": None
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, "UNKNOWN"),
    "objective_value": objective_value,
    "example_output": output
}))