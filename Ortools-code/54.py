import json
from ortools.linear_solver import pywraplp

inp = {
    "products": [
        {
            "name": "I",
            "unit_profit_thousand_yuan": 3,
            "equipment_hours": {
                "A": 8,
                "B": 10,
                "C": 2
            }
        },
        {
            "name": "II",
            "unit_profit_thousand_yuan": 2,
            "equipment_hours": {
                "A": 2,
                "B": 5,
                "C": 13
            }
        },
        {
            "name": "III",
            "unit_profit_thousand_yuan": 2.9,
            "equipment_hours": {
                "A": 10,
                "B": 8,
                "C": 10
            }
        }
    ],
    "equipment_capacity_hours": {
        "A": 300,
        "B": 400,
        "C": 420
    }
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver")

products = inp["products"]
equipments = list(inp["equipment_capacity_hours"].keys())

x = {
    p["name"]: solver.NumVar(0.0, solver.infinity(), f"x_{p['name']}")
    for p in products
}

for e in equipments:
    solver.Add(
        sum(p["equipment_hours"][e] * x[p["name"]] for p in products)
        <= inp["equipment_capacity_hours"][e]
    )

objective = solver.Objective()
for p in products:
    objective.SetCoefficient(x[p["name"]], p["unit_profit_thousand_yuan"])
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

def r(v):
    return round(float(v), 6)

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    production_plan = {p["name"]: r(x[p["name"]].solution_value()) for p in products}
    equipment_usage_hours = {}
    unused_capacity_hours = {}
    for e in equipments:
        used = sum(p["equipment_hours"][e] * x[p["name"]].solution_value() for p in products)
        cap = inp["equipment_capacity_hours"][e]
        equipment_usage_hours[e] = r(used)
        unused_capacity_hours[e] = r(cap - used)

    output = {
        "production_plan": production_plan,
        "equipment_usage_hours": equipment_usage_hours,
        "unused_capacity_hours": unused_capacity_hours,
        "max_profit_thousand_yuan": r(objective.Value())
    }
    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": r(objective.Value()),
        "example_output": output
    }
else:
    output = {
        "production_plan": {"I": 0, "II": 0, "III": 0},
        "equipment_usage_hours": {"A": 0, "B": 0, "C": 0},
        "unused_capacity_hours": {
            "A": inp["equipment_capacity_hours"]["A"],
            "B": inp["equipment_capacity_hours"]["B"],
            "C": inp["equipment_capacity_hours"]["C"]
        },
        "max_profit_thousand_yuan": 0
    }
    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))