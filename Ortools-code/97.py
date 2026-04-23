import json
from ortools.linear_solver import pywraplp

inp = {
    "raw_materials": [
        {
            "name": "raw_A",
            "sulfur_fraction": 0.03,
            "purchase_cost_thousand_yuan_per_ton": 6.0,
            "max_supply_tons": None
        },
        {
            "name": "raw_B",
            "sulfur_fraction": 0.01,
            "purchase_cost_thousand_yuan_per_ton": 16.0,
            "max_supply_tons": None
        },
        {
            "name": "raw_C",
            "sulfur_fraction": 0.02,
            "purchase_cost_thousand_yuan_per_ton": 10.0,
            "max_supply_tons": None
        },
        {
            "name": "raw_D",
            "sulfur_fraction": 0.01,
            "purchase_cost_thousand_yuan_per_ton": 15.0,
            "max_supply_tons": 50.0
        }
    ],
    "products": [
        {
            "name": "product_A",
            "max_sulfur_fraction": 0.025,
            "selling_price_thousand_yuan_per_ton": 9.15,
            "max_demand_tons": 100.0
        },
        {
            "name": "product_B",
            "max_sulfur_fraction": 0.015,
            "selling_price_thousand_yuan_per_ton": 9.15,
            "max_demand_tons": 200.0
        }
    ],
    "process": {
        "first_stage_mix": ["raw_A", "raw_B", "raw_D"],
        "second_stage_added_raw": "raw_C"
    }
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver.")

raws = inp["raw_materials"]
products = inp["products"]

raw_names = [r["name"] for r in raws]
prod_names = [p["name"] for p in products]

raw_sulfur = {r["name"]: r["sulfur_fraction"] for r in raws}
raw_cost = {r["name"]: r["purchase_cost_thousand_yuan_per_ton"] for r in raws}
raw_supply = {r["name"]: r["max_supply_tons"] for r in raws}

prod_sulfur_max = {p["name"]: p["max_sulfur_fraction"] for p in products}
prod_price = {p["name"]: p["selling_price_thousand_yuan_per_ton"] for p in products}
prod_demand = {p["name"]: p["max_demand_tons"] for p in products}

x = {}
for r in raw_names:
    for p in prod_names:
        x[(r, p)] = solver.NumVar(0.0, solver.infinity(), f"x_{r}_{p}")

# Demand limits
for p in prod_names:
    solver.Add(solver.Sum(x[(r, p)] for r in raw_names) <= prod_demand[p])

# Supply limits
for r in raw_names:
    if raw_supply[r] is not None:
        solver.Add(solver.Sum(x[(r, p)] for p in prod_names) <= raw_supply[r])

# Sulfur quality constraints
for p in prod_names:
    total_product = solver.Sum(x[(r, p)] for r in raw_names)
    total_sulfur = solver.Sum(raw_sulfur[r] * x[(r, p)] for r in raw_names)
    solver.Add(total_sulfur <= prod_sulfur_max[p] * total_product)

# Objective: maximize profit = revenue - raw material purchase cost
objective = solver.Objective()
for r in raw_names:
    for p in prod_names:
        objective.SetCoefficient(x[(r, p)], prod_price[p] - raw_cost[r])
objective.SetMaximization()

status_code = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}
status = status_map.get(status_code, str(status_code))

def clean(v, tol=1e-9):
    if abs(v) < tol:
        return 0.0
    return round(v, 6)

product_output_tons = {}
raw_allocation_tons = {}

for p in prod_names:
    allocations = {}
    total = 0.0
    for r in raw_names:
        val = x[(r, p)].solution_value() if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else 0.0
        val = clean(val)
        allocations[r] = val
        total += val
    raw_allocation_tons[p] = allocations
    product_output_tons[p] = clean(total)

objective_value = clean(solver.Objective().Value()) if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else None

output = {
    "product_output_tons": {
        "product_A": product_output_tons.get("product_A", 0.0),
        "product_B": product_output_tons.get("product_B", 0.0)
    },
    "raw_allocation_tons": {
        "product_A": raw_allocation_tons.get("product_A", {"raw_A": 0.0, "raw_B": 0.0, "raw_C": 0.0, "raw_D": 0.0}),
        "product_B": raw_allocation_tons.get("product_B", {"raw_A": 0.0, "raw_B": 0.0, "raw_C": 0.0, "raw_D": 0.0})
    },
    "total_profit_thousand_yuan": objective_value if objective_value is not None else 0.0
}

print(json.dumps({
    "status": status,
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))