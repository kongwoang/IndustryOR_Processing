import json
from ortools.linear_solver import pywraplp

inp = {
    "num_furnaces": 2,
    "time_limit_hours_per_furnace": 12,
    "methods": [
        {
            "name": "method_1",
            "time_hours_per_run": 2,
            "fuel_cost_per_run": 50,
            "steel_tons_per_run": 10
        },
        {
            "name": "method_2",
            "time_hours_per_run": 3,
            "fuel_cost_per_run": 70,
            "steel_tons_per_run": 10
        }
    ],
    "min_total_steel_tons": 30
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver.")

methods = inp["methods"]
total_available_furnace_hours = inp["num_furnaces"] * inp["time_limit_hours_per_furnace"]

x = {}
for i, method in enumerate(methods):
    x[i] = solver.NumVar(0.0, solver.infinity(), f"x_{method['name']}")

solver.Add(
    sum(methods[i]["time_hours_per_run"] * x[i] for i in range(len(methods)))
    <= total_available_furnace_hours
)

solver.Add(
    sum(methods[i]["steel_tons_per_run"] * x[i] for i in range(len(methods)))
    >= inp["min_total_steel_tons"]
)

objective = solver.Objective()
for i in range(len(methods)):
    objective.SetCoefficient(x[i], methods[i]["fuel_cost_per_run"])
objective.SetMinimization()

status_code = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}
status = status_map.get(status_code, "UNKNOWN")

def clean_number(v):
    if v is None:
        return None
    if abs(v) < 1e-9:
        v = 0.0
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    return float(v)

if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    method_allocations = []
    total_furnace_hours_used = 0.0
    total_steel_tons = 0.0
    total_fuel_cost = 0.0

    for i, method in enumerate(methods):
        runs = x[i].solution_value()
        method_allocations.append({
            "name": method["name"],
            "runs": clean_number(runs)
        })
        total_furnace_hours_used += method["time_hours_per_run"] * runs
        total_steel_tons += method["steel_tons_per_run"] * runs
        total_fuel_cost += method["fuel_cost_per_run"] * runs

    output = {
        "method_allocations": method_allocations,
        "total_furnace_hours_used": clean_number(total_furnace_hours_used),
        "total_steel_tons": clean_number(total_steel_tons),
        "total_fuel_cost": clean_number(total_fuel_cost)
    }
    objective_value = clean_number(solver.Objective().Value())
else:
    output = {
        "method_allocations": [],
        "total_furnace_hours_used": None,
        "total_steel_tons": None,
        "total_fuel_cost": None
    }
    objective_value = None

result = {
    "status": status,
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result))