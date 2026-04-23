import json
from ortools.linear_solver import pywraplp

inp = {
    "products": {
        "robots": {
            "profit": 15,
            "plastic": 30,
            "electronic_components": 8
        },
        "model_cars": {
            "profit": 8,
            "plastic": 10,
            "electronic_components": 5
        },
        "building_blocks": {
            "profit": 12,
            "plastic": 20,
            "electronic_components": 3
        },
        "dolls": {
            "profit": 5,
            "plastic": 15,
            "electronic_components": 2
        }
    },
    "resource_limits": {
        "plastic": 1200,
        "electronic_components": 800
    },
    "logical_constraints": {
        "robots_and_dolls_mutually_exclusive_if_manufactured": True,
        "model_cars_imply_building_blocks_if_manufactured": True,
        "dolls_quantity_at_most_model_cars_quantity": True
    }
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

products = list(inp["products"].keys())
resources = list(inp["resource_limits"].keys())

upper_bounds = {}
for p in products:
    bounds = []
    for r in resources:
        usage = inp["products"][p][r]
        if usage > 0:
            bounds.append(inp["resource_limits"][r] // usage)
    upper_bounds[p] = min(bounds)

q = {p: solver.IntVar(0, upper_bounds[p], f"q_{p}") for p in products}
y = {p: solver.IntVar(0, 1, f"y_{p}") for p in products}

for p in products:
    solver.Add(q[p] <= upper_bounds[p] * y[p])
    solver.Add(q[p] >= y[p])

for r in resources:
    solver.Add(
        sum(inp["products"][p][r] * q[p] for p in products) <= inp["resource_limits"][r]
    )

if inp["logical_constraints"]["robots_and_dolls_mutually_exclusive_if_manufactured"]:
    solver.Add(y["robots"] + y["dolls"] <= 1)

if inp["logical_constraints"]["model_cars_imply_building_blocks_if_manufactured"]:
    solver.Add(y["model_cars"] <= y["building_blocks"])

if inp["logical_constraints"]["dolls_quantity_at_most_model_cars_quantity"]:
    solver.Add(q["dolls"] <= q["model_cars"])

solver.Maximize(sum(inp["products"][p]["profit"] * q[p] for p in products))

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

def clean_number(x):
    if x is None:
        return None
    if abs(x - round(x)) <= 1e-9:
        return int(round(x))
    return x

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "production_quantities": {
            p: int(round(q[p].solution_value())) for p in products
        },
        "manufactured_types": {
            p: int(round(y[p].solution_value())) for p in products
        }
    }
    objective_value = clean_number(solver.Objective().Value())
else:
    output = {
        "production_quantities": {
            "robots": 0,
            "model_cars": 0,
            "building_blocks": 0,
            "dolls": 0
        },
        "manufactured_types": {
            "robots": 0,
            "model_cars": 0,
            "building_blocks": 0,
            "dolls": 0
        }
    }
    objective_value = None

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result))