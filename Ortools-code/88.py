import json
from ortools.linear_solver import pywraplp

inp = {
    "products": ["liquid", "solid"],
    "machines": ["Machine 1", "Machine 2"],
    "processing_time_minutes_per_lot": {
        "liquid": {
            "Machine 1": 50,
            "Machine 2": 30
        },
        "solid": {
            "Machine 1": 24,
            "Machine 2": 33
        }
    },
    "beginning_inventory_lots": {
        "liquid": 30,
        "solid": 90
    },
    "available_hours": {
        "Machine 1": 40,
        "Machine 2": 35
    },
    "forecast_demand_lots": {
        "liquid": 75,
        "solid": 95
    },
    "fractional_lots_allowed": True
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver.")

products = inp["products"]
machines = inp["machines"]

production = {
    p: solver.NumVar(0.0, solver.infinity(), f"production_{p}")
    for p in products
}

ending_inventory = {
    p: solver.NumVar(0.0, solver.infinity(), f"ending_inventory_{p}")
    for p in products
}

for m in machines:
    solver.Add(
        sum(inp["processing_time_minutes_per_lot"][p][m] * production[p] for p in products)
        <= inp["available_hours"][m] * 60
    )

for p in products:
    solver.Add(
        ending_inventory[p]
        == inp["beginning_inventory_lots"][p] + production[p] - inp["forecast_demand_lots"][p]
    )

solver.Maximize(sum(ending_inventory[p] for p in products))

status = solver.Solve()

def clean_number(x):
    if abs(x - round(x)) <= 1e-9:
        return int(round(x))
    return round(x, 10)

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
        "production_lots": {
            p: clean_number(production[p].solution_value())
            for p in products
        },
        "ending_inventory_lots": {
            p: clean_number(ending_inventory[p].solution_value())
            for p in products
        },
        "total_ending_inventory_lots": clean_number(
            sum(ending_inventory[p].solution_value() for p in products)
        )
    }
    objective_value = output["total_ending_inventory_lots"]
else:
    output = {
        "production_lots": {p: None for p in products},
        "ending_inventory_lots": {p: None for p in products},
        "total_ending_inventory_lots": None
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, "UNKNOWN"),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))