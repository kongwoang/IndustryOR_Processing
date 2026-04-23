import json
from ortools.linear_solver import pywraplp

inp = {
    "workshops": [
        {
            "name": "A",
            "capacity_hours": 100,
            "production_rates": {
                "Component 1": 10,
                "Component 2": 15,
                "Component 3": 5
            }
        },
        {
            "name": "B",
            "capacity_hours": 150,
            "production_rates": {
                "Component 1": 15,
                "Component 2": 10,
                "Component 3": 5
            }
        },
        {
            "name": "C",
            "capacity_hours": 80,
            "production_rates": {
                "Component 1": 20,
                "Component 2": 5,
                "Component 3": 10
            }
        },
        {
            "name": "D",
            "capacity_hours": 200,
            "production_rates": {
                "Component 1": 10,
                "Component 2": 15,
                "Component 3": 20
            }
        }
    ],
    "components": ["Component 1", "Component 2", "Component 3"],
    "required_units_per_product": {
        "Component 1": 1,
        "Component 2": 1,
        "Component 3": 1
    },
    "objective": "maximize_completed_products"
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is unavailable.")

workshops = [w["name"] for w in inp["workshops"]]
components = inp["components"]
capacity = {w["name"]: w["capacity_hours"] for w in inp["workshops"]}
rate = {
    w["name"]: {c: w["production_rates"][c] for c in components}
    for w in inp["workshops"]
}

# Hours allocated by workshop w to component c
x = {}
for w in workshops:
    for c in components:
        x[(w, c)] = solver.NumVar(0.0, solver.infinity(), f"x_{w}_{c}")

# Number of completed products must be integer
completed_products = solver.IntVar(0, solver.infinity(), "completed_products")

# Capacity constraints for each workshop
for w in workshops:
    solver.Add(sum(x[(w, c)] for c in components) <= capacity[w])

# To complete P products, each component must be produced in at least P units
for c in components:
    solver.Add(
        sum(rate[w][c] * x[(w, c)] for w in workshops)
        >= inp["required_units_per_product"][c] * completed_products
    )

solver.Maximize(completed_products)
status_code = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
}
status = status_map.get(status_code, str(status_code))

def clean(v):
    if abs(v) < 1e-9:
        v = 0.0
    return round(v, 6)

if status in {"OPTIMAL", "FEASIBLE"}:
    objective_value = clean(solver.Objective().Value())

    hours_allocated = {
        w: {c: clean(x[(w, c)].solution_value()) for c in components}
        for w in workshops
    }

    component_units_produced = {
        c: clean(sum(rate[w][c] * x[(w, c)].solution_value() for w in workshops))
        for c in components
    }

    hours_used_by_workshop = {
        w: clean(sum(x[(w, c)].solution_value() for c in components))
        for w in workshops
    }

    output = {
        "completed_products": objective_value,
        "hours_allocated": hours_allocated,
        "component_units_produced": component_units_produced,
        "hours_used_by_workshop": hours_used_by_workshop
    }
else:
    objective_value = None
    output = {
        "completed_products": None,
        "hours_allocated": {},
        "component_units_produced": {},
        "hours_used_by_workshop": {}
    }

print(json.dumps({
    "status": status,
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))