import json
from ortools.linear_solver import pywraplp

inp = {
    "products": [
        {
            "name": "A1",
            "kg_per_barrel": 3,
            "hours_per_barrel": 12,
            "profit_per_kg": 24,
            "daily_max_kg": 100
        },
        {
            "name": "A2",
            "kg_per_barrel": 4,
            "hours_per_barrel": 8,
            "profit_per_kg": 16,
            "daily_max_kg": None
        }
    ],
    "milk_supply_barrels_per_day": 50,
    "labor_hours_per_day": 480,
    "objective": "maximize_daily_profit"
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver")

products = {p["name"]: p for p in inp["products"]}

# Decision variables: barrels of milk allocated to each product
x = {
    name: solver.NumVar(0.0, solver.infinity(), f"barrels_{name}")
    for name in products
}

# Milk supply constraint
solver.Add(sum(x[name] for name in products) <= inp["milk_supply_barrels_per_day"])

# Labor hours constraint
solver.Add(
    sum(products[name]["hours_per_barrel"] * x[name] for name in products)
    <= inp["labor_hours_per_day"]
)

# Product-specific daily capacity constraints
for name, p in products.items():
    if p["daily_max_kg"] is not None:
        solver.Add(p["kg_per_barrel"] * x[name] <= p["daily_max_kg"])

# Objective: maximize daily profit
objective = solver.Objective()
for name, p in products.items():
    profit_per_barrel = p["kg_per_barrel"] * p["profit_per_kg"]
    objective.SetCoefficient(x[name], profit_per_barrel)
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
        "production_barrels": {
            name: x[name].solution_value() for name in products
        },
        "production_kg": {
            name: products[name]["kg_per_barrel"] * x[name].solution_value()
            for name in products
        }
    }
    objective_value = objective.Value()
else:
    output = {
        "production_barrels": {
            name: None for name in products
        },
        "production_kg": {
            name: None for name in products
        }
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))