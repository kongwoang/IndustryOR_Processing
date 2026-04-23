import json
from ortools.linear_solver import pywraplp

inp = {
    "raw_materials": ["A", "B", "C"],
    "candy_brands": ["A", "B", "C"],
    "raw_material_cost": {
        "A": 2.0,
        "B": 1.5,
        "C": 1.0
    },
    "monthly_limit": {
        "A": 2000.0,
        "B": 2500.0,
        "C": 1200.0
    },
    "processing_fee": {
        "A": 0.5,
        "B": 0.4,
        "C": 0.3
    },
    "selling_price": {
        "A": 3.4,
        "B": 2.85,
        "C": 2.25
    },
    "min_percent": {
        "A": {
            "A": 0.6
        },
        "B": {
            "A": 0.15
        },
        "C": {}
    },
    "max_percent": {
        "A": {
            "C": 0.2
        },
        "B": {
            "C": 0.6
        },
        "C": {
            "C": 0.5
        }
    }
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("GLOP solver is unavailable.")

x = {}
for r in inp["raw_materials"]:
    for b in inp["candy_brands"]:
        x[(r, b)] = solver.NumVar(0.0, solver.infinity(), f"x_{r}_{b}")

y = {}
for b in inp["candy_brands"]:
    y[b] = solver.NumVar(0.0, solver.infinity(), f"y_{b}")

for b in inp["candy_brands"]:
    solver.Add(y[b] == sum(x[(r, b)] for r in inp["raw_materials"]))

for b, rules in inp["min_percent"].items():
    for r, pct in rules.items():
        solver.Add(x[(r, b)] >= pct * y[b])

for b, rules in inp["max_percent"].items():
    for r, pct in rules.items():
        solver.Add(x[(r, b)] <= pct * y[b])

for r in inp["raw_materials"]:
    solver.Add(sum(x[(r, b)] for b in inp["candy_brands"]) <= inp["monthly_limit"][r])

objective = solver.Objective()
for b in inp["candy_brands"]:
    objective.SetCoefficient(y[b], inp["selling_price"][b] - inp["processing_fee"][b])
for r in inp["raw_materials"]:
    for b in inp["candy_brands"]:
        objective.SetCoefficient(x[(r, b)], -inp["raw_material_cost"][r])
objective.SetMaximization()

status_code = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
}

def clean(v):
    if abs(v) < 1e-9:
        return 0.0
    return round(v, 6)

output = {
    "production_kg": {
        b: clean(y[b].solution_value()) if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else None
        for b in inp["candy_brands"]
    },
    "raw_material_allocation_kg": {
        b: {
            r: clean(x[(r, b)].solution_value()) if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else None
            for r in inp["raw_materials"]
        }
        for b in inp["candy_brands"]
    }
}

result = {
    "status": status_map.get(status_code, str(status_code)),
    "objective_value": clean(solver.Objective().Value()) if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else None,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))