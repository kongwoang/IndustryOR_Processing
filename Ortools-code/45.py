import json
from ortools.linear_solver import pywraplp

inp = {
    "resources": {
        "technical_preparation_hours": 100,
        "labor_hours": 700,
        "materials_kg": 400
    },
    "products": {
        "A": {
            "resource_usage_per_unit": {
                "technical_preparation_hours": 1,
                "labor_hours": 10,
                "materials_kg": 3
            },
            "profit_tiers": [
                {"min_units": 0, "max_units": 40, "unit_profit": 10},
                {"min_units": 40, "max_units": 100, "unit_profit": 9},
                {"min_units": 100, "max_units": 150, "unit_profit": 8},
                {"min_units": 150, "max_units": None, "unit_profit": 7}
            ]
        },
        "B": {
            "resource_usage_per_unit": {
                "technical_preparation_hours": 2,
                "labor_hours": 4,
                "materials_kg": 2
            },
            "profit_tiers": [
                {"min_units": 0, "max_units": 50, "unit_profit": 6},
                {"min_units": 50, "max_units": 100, "unit_profit": 4},
                {"min_units": 100, "max_units": None, "unit_profit": 3}
            ]
        },
        "C": {
            "resource_usage_per_unit": {
                "technical_preparation_hours": 1,
                "labor_hours": 5,
                "materials_kg": 1
            },
            "profit_tiers": [
                {"min_units": 0, "max_units": 100, "unit_profit": 5},
                {"min_units": 100, "max_units": None, "unit_profit": 4}
            ]
        }
    }
}


def normalize_number(x):
    if x is None:
        return None
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    return x


solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    solver = pywraplp.Solver.CreateSolver("CBC")

products = list(inp["products"].keys())
resources = list(inp["resources"].keys())

max_units = {}
for p in products:
    usage = inp["products"][p]["resource_usage_per_unit"]
    max_units[p] = min(
        inp["resources"][r] // usage[r]
        for r in resources
        if usage[r] > 0
    )

q = {}
x = {}
y = {}

objective = solver.Objective()
objective.SetMaximization()

for p in products:
    q[p] = solver.IntVar(0, int(max_units[p]), f"q_{p}")
    active_tiers = []

    for t, tier in enumerate(inp["products"][p]["profit_tiers"]):
        lb = int(tier["min_units"])
        ub = int(max_units[p] if tier["max_units"] is None else min(tier["max_units"], max_units[p]))

        if ub < lb:
            continue

        x[p, t] = solver.IntVar(0, ub, f"x_{p}_{t}")
        y[p, t] = solver.BoolVar(f"y_{p}_{t}")

        solver.Add(x[p, t] <= ub * y[p, t])
        if lb > 0:
            solver.Add(x[p, t] >= lb * y[p, t])

        active_tiers.append(t)
        objective.SetCoefficient(x[p, t], tier["unit_profit"])

    solver.Add(q[p] == sum(x[p, t] for t in active_tiers))
    solver.Add(sum(y[p, t] for t in active_tiers) <= 1)

for r in resources:
    solver.Add(
        sum(inp["products"][p]["resource_usage_per_unit"][r] * q[p] for p in products)
        <= inp["resources"][r]
    )

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}
status_str = status_map.get(status, str(status))

output = {
    "production_quantities": {p: 0 for p in products},
    "selected_profit_tiers": {p: None for p in products}
}

objective_value = None

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output["production_quantities"] = {
        p: normalize_number(q[p].solution_value()) for p in products
    }

    for p in products:
        selected = None
        for t, tier in enumerate(inp["products"][p]["profit_tiers"]):
            if (p, t) in y and y[p, t].solution_value() > 0.5:
                if tier["max_units"] is None:
                    selected = f"{tier['min_units']}+"
                else:
                    selected = f"{tier['min_units']}-{tier['max_units']}"
                break
        output["selected_profit_tiers"][p] = selected

    objective_value = normalize_number(objective.Value())

print(json.dumps({
    "status": status_str,
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))