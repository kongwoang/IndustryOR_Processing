# Source: :contentReference[oaicite:0]{index=0}
import json
from ortools.linear_solver import pywraplp

inp = {
    "products": [
        {
            "name": "A",
            "profit_per_unit": 300,
            "process_hours": {
                "I": 4,
                "II": 3
            },
            "minimum_units": 10
        },
        {
            "name": "B",
            "profit_per_unit": 450,
            "process_hours": {
                "I": 6,
                "II": 2
            },
            "minimum_units": 15
        }
    ],
    "process_capacities": {
        "I": 150,
        "II": 70
    },
    "minimum_total_profit": 10000
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    solver = pywraplp.Solver.CreateSolver("CBC")
if solver is None:
    raise RuntimeError("No suitable MIP solver is available in OR-Tools.")

x = {}
for p in inp["products"]:
    name = p["name"]
    x[name] = solver.IntVar(0, solver.infinity(), f"x_{name}")

for p in inp["products"]:
    solver.Add(x[p["name"]] >= p["minimum_units"])

solver.Add(
    solver.Sum(p["process_hours"]["I"] * x[p["name"]] for p in inp["products"])
    <= inp["process_capacities"]["I"]
)
solver.Add(
    solver.Sum(p["process_hours"]["II"] * x[p["name"]] for p in inp["products"])
    <= inp["process_capacities"]["II"]
)

profit_expr = solver.Sum(p["profit_per_unit"] * x[p["name"]] for p in inp["products"])
solver.Add(profit_expr >= inp["minimum_total_profit"])

solver.Maximize(profit_expr)

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
    production = {p["name"]: int(round(x[p["name"]].solution_value())) for p in inp["products"]}
    process_time_used = {
        "I": sum(p["process_hours"]["I"] * production[p["name"]] for p in inp["products"]),
        "II": sum(p["process_hours"]["II"] * production[p["name"]] for p in inp["products"])
    }
    total_profit = sum(p["profit_per_unit"] * production[p["name"]] for p in inp["products"])

    output = {
        "production": production,
        "process_time_used": process_time_used,
        "total_profit": total_profit
    }

    objective_value = int(round(solver.Objective().Value()))
else:
    output = {
        "production": {
            "A": None,
            "B": None
        },
        "process_time_used": {
            "I": None,
            "II": None
        },
        "total_profit": None
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))