import json
from ortools.linear_solver import pywraplp

inp = {
    "protein_options": [
        {
            "name": "chicken",
            "protein_per_100g": 23.0,
            "cost_per_100g": 3.0
        },
        {
            "name": "salmon",
            "protein_per_100g": 20.0,
            "cost_per_100g": 5.0
        },
        {
            "name": "tofu",
            "protein_per_100g": 8.0,
            "cost_per_100g": 1.5
        }
    ],
    "vegetable_options": [
        {
            "name": "broccoli",
            "protein_per_pack": 2.8,
            "cost_per_pack": 1.2,
            "weight_per_pack_g": 100
        },
        {
            "name": "carrots",
            "protein_per_pack": 0.9,
            "cost_per_pack": 0.8,
            "weight_per_pack_g": 100
        },
        {
            "name": "spinach",
            "protein_per_pack": 2.9,
            "cost_per_pack": 1.5,
            "weight_per_pack_g": 100
        },
        {
            "name": "bell_pepper",
            "protein_per_pack": 1.0,
            "cost_per_pack": 1.0,
            "weight_per_pack_g": 100
        },
        {
            "name": "mushrooms",
            "protein_per_pack": 3.1,
            "cost_per_pack": 2.0,
            "weight_per_pack_g": 100
        }
    ],
    "constraints": {
        "budget_max": 20.0,
        "weight_max_g": 800,
        "min_distinct_vegetables": 3
    }
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

protein_items = inp["protein_options"]
veg_items = inp["vegetable_options"]
budget_max = inp["constraints"]["budget_max"]
weight_max_g = inp["constraints"]["weight_max_g"]
min_distinct_vegetables = inp["constraints"]["min_distinct_vegetables"]

max_veg_packs_per_type = weight_max_g // 100

# Decision variables
x = {}  # continuous 100g units for proteins
for item in protein_items:
    x[item["name"]] = solver.NumVar(0.0, solver.infinity(), f'x_{item["name"]}')

v = {}  # integer number of 100g packs for vegetables
y = {}  # binary selection for vegetable type
for item in veg_items:
    name = item["name"]
    v[name] = solver.IntVar(0, max_veg_packs_per_type, f"v_{name}")
    y[name] = solver.IntVar(0, 1, f"y_{name}")

# Budget constraint
solver.Add(
    sum(x[item["name"]] * item["cost_per_100g"] for item in protein_items) +
    sum(v[item["name"]] * item["cost_per_pack"] for item in veg_items)
    <= budget_max
)

# Weight constraint
solver.Add(
    sum(x[item["name"]] * 100.0 for item in protein_items) +
    sum(v[item["name"]] * item["weight_per_pack_g"] for item in veg_items)
    <= weight_max_g
)

# At least three different vegetable types
solver.Add(sum(y[item["name"]] for item in veg_items) >= min_distinct_vegetables)

# Linking: if selected, at least one pack; if not selected, zero packs
for item in veg_items:
    name = item["name"]
    solver.Add(v[name] >= y[name])
    solver.Add(v[name] <= max_veg_packs_per_type * y[name])

# Objective: maximize total protein
objective = solver.Objective()
for item in protein_items:
    objective.SetCoefficient(x[item["name"]], item["protein_per_100g"])
for item in veg_items:
    objective.SetCoefficient(v[item["name"]], item["protein_per_pack"])
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
    protein_amounts_100g = {}
    for item in protein_items:
        val = x[item["name"]].solution_value()
        protein_amounts_100g[item["name"]] = round(val, 6)

    vegetable_packs = {}
    for item in veg_items:
        val = int(round(v[item["name"]].solution_value()))
        vegetable_packs[item["name"]] = val

    total_cost = sum(
        protein_amounts_100g[item["name"]] * item["cost_per_100g"] for item in protein_items
    ) + sum(
        vegetable_packs[item["name"]] * item["cost_per_pack"] for item in veg_items
    )

    total_weight_g = sum(
        protein_amounts_100g[item["name"]] * 100.0 for item in protein_items
    ) + sum(
        vegetable_packs[item["name"]] * item["weight_per_pack_g"] for item in veg_items
    )

    total_protein_g = sum(
        protein_amounts_100g[item["name"]] * item["protein_per_100g"] for item in protein_items
    ) + sum(
        vegetable_packs[item["name"]] * item["protein_per_pack"] for item in veg_items
    )

    output = {
        "protein_amounts_100g": protein_amounts_100g,
        "vegetable_packs": vegetable_packs,
        "total_cost": round(total_cost, 6),
        "total_weight_g": round(total_weight_g, 6),
        "total_protein_g": round(total_protein_g, 6)
    }

    objective_value = round(solver.Objective().Value(), 6)
else:
    output = {
        "protein_amounts_100g": {
            "chicken": 0.0,
            "salmon": 0.0,
            "tofu": 0.0
        },
        "vegetable_packs": {
            "broccoli": 0,
            "carrots": 0,
            "spinach": 0,
            "bell_pepper": 0,
            "mushrooms": 0
        },
        "total_cost": 0.0,
        "total_weight_g": 0.0,
        "total_protein_g": 0.0
    }
    objective_value = None

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))