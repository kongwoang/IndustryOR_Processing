# Source: :contentReference[oaicite:0]{index=0}
import json
from ortools.linear_solver import pywraplp

inp = {
    "animal_types": ["cows", "sheep", "chickens"],
    "selling_price": {
        "cows": 500,
        "sheep": 200,
        "chickens": 8
    },
    "feed_cost": {
        "cows": 100,
        "sheep": 80,
        "chickens": 5
    },
    "manure_per_day": {
        "cows": 10,
        "sheep": 5,
        "chickens": 3
    },
    "max_manure": 800,
    "max_chickens": 50,
    "min_cows": 10,
    "min_sheep": 20,
    "max_total_animals": 100
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

x = {
    animal: solver.IntVar(0, solver.infinity(), animal)
    for animal in inp["animal_types"]
}

unit_profit = {
    animal: inp["selling_price"][animal] - inp["feed_cost"][animal]
    for animal in inp["animal_types"]
}

solver.Add(
    sum(inp["manure_per_day"][animal] * x[animal] for animal in inp["animal_types"])
    <= inp["max_manure"]
)
solver.Add(x["chickens"] <= inp["max_chickens"])
solver.Add(x["cows"] >= inp["min_cows"])
solver.Add(x["sheep"] >= inp["min_sheep"])
solver.Add(sum(x[animal] for animal in inp["animal_types"]) <= inp["max_total_animals"])

objective = solver.Objective()
for animal in inp["animal_types"]:
    objective.SetCoefficient(x[animal], unit_profit[animal])
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

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "cows": int(round(x["cows"].solution_value())),
        "sheep": int(round(x["sheep"].solution_value())),
        "chickens": int(round(x["chickens"].solution_value()))
    }
    objective_value = int(round(solver.Objective().Value()))
else:
    output = {
        "cows": 0,
        "sheep": 0,
        "chickens": 0
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}))