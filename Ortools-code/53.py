# Source: :contentReference[oaicite:0]{index=0}
import json
from ortools.linear_solver import pywraplp

inp = {
    "children": [
        {"name": "Harry", "cost": 1200},
        {"name": "Hermione", "cost": 1650},
        {"name": "Ron", "cost": 750},
        {"name": "Fred", "cost": 800},
        {"name": "George", "cost": 800},
        {"name": "Ginny", "cost": 1500}
    ],
    "max_children": 4,
    "min_children": 3,
    "fixed_take": ["Ginny"],
    "incompatible_pairs": [
        ["Harry", "Fred"],
        ["Harry", "George"]
    ],
    "implications": [
        {"if_taken": "George", "must_also_take": ["Fred", "Hermione"]}
    ]
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("Failed to create SCIP solver.")

children = [c["name"] for c in inp["children"]]
cost = {c["name"]: c["cost"] for c in inp["children"]}

x = {name: solver.BoolVar(f"x_{name}") for name in children}

solver.Add(solver.Sum(x[name] for name in children) <= inp["max_children"])
solver.Add(solver.Sum(x[name] for name in children) >= inp["min_children"])

for name in inp["fixed_take"]:
    solver.Add(x[name] == 1)

for a, b in inp["incompatible_pairs"]:
    solver.Add(x[a] + x[b] <= 1)

for rule in inp["implications"]:
    src = rule["if_taken"]
    for dst in rule["must_also_take"]:
        solver.Add(x[src] <= x[dst])

solver.Minimize(solver.Sum(cost[name] * x[name] for name in children))

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
    selected_children = [name for name in children if x[name].solution_value() > 0.5]
    total_cost = sum(cost[name] for name in selected_children)
    output = {
        "selected_children": selected_children,
        "total_cost": total_cost
    }
    objective_value = solver.Objective().Value()
else:
    output = {
        "selected_children": [],
        "total_cost": None
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}))