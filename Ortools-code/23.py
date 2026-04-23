import json
from ortools.linear_solver import pywraplp

inp = {
    "products": [
        {
            "name": "shirt",
            "labor_per_unit": 3,
            "material_per_unit": 4,
            "selling_price": 120,
            "variable_cost": 60
        },
        {
            "name": "short_sleeve",
            "labor_per_unit": 2,
            "material_per_unit": 3,
            "selling_price": 80,
            "variable_cost": 40
        },
        {
            "name": "casual_cloth",
            "labor_per_unit": 6,
            "material_per_unit": 6,
            "selling_price": 180,
            "variable_cost": 80
        }
    ],
    "resources": {
        "available_labor": 1500,
        "available_material": 1600
    },
    "weekly_fixed_costs": [2000, 1500, 1000]
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("GLOP solver is not available.")

products = inp["products"]
labor_cap = inp["resources"]["available_labor"]
material_cap = inp["resources"]["available_material"]
total_fixed_cost = sum(inp["weekly_fixed_costs"])

x = {}
for p in products:
    x[p["name"]] = solver.NumVar(0.0, solver.infinity(), f"x_{p['name']}")

solver.Add(sum(p["labor_per_unit"] * x[p["name"]] for p in products) <= labor_cap)
solver.Add(sum(p["material_per_unit"] * x[p["name"]] for p in products) <= material_cap)

objective = solver.Objective()
for p in products:
    name = p["name"]
    contribution_margin = p["selling_price"] - p["variable_cost"]
    objective.SetCoefficient(x[name], contribution_margin)
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
    production_plan = []
    total_revenue = 0.0
    total_variable_cost = 0.0

    for p in products:
        name = p["name"]
        qty = x[name].solution_value()
        production_plan.append({
            "name": name,
            "produce": qty
        })
        total_revenue += p["selling_price"] * qty
        total_variable_cost += p["variable_cost"] * qty

    output = {
        "production_plan": production_plan,
        "total_revenue": total_revenue,
        "total_variable_cost": total_variable_cost,
        "total_fixed_cost": total_fixed_cost,
        "total_profit": total_revenue - total_variable_cost - total_fixed_cost
    }

    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": output["total_profit"],
        "example_output": output
    }
else:
    output = {
        "production_plan": [],
        "total_revenue": 0.0,
        "total_variable_cost": 0.0,
        "total_fixed_cost": total_fixed_cost,
        "total_profit": None
    }
    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))