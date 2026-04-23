import json
from ortools.linear_solver import pywraplp

inp = {
    "children": [
        {
            "name": "Alice",
            "cost": 1000
        },
        {
            "name": "Bob",
            "cost": 900
        },
        {
            "name": "Charlie",
            "cost": 600
        },
        {
            "name": "Diana",
            "cost": 500
        },
        {
            "name": "Ella",
            "cost": 700
        }
    ],
    "max_children": 3,
    "min_children": 2,
    "must_take": [
        "Bob"
    ],
    "incompatible_pairs": [
        [
            "Alice",
            "Diana"
        ],
        [
            "Bob",
            "Charlie"
        ]
    ],
    "implications": [
        {
            "if_taken": "Charlie",
            "must_also_take": "Diana"
        },
        {
            "if_taken": "Diana",
            "must_also_take": "Ella"
        }
    ]
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

children = inp["children"]
names = [c["name"] for c in children]
costs = {c["name"]: c["cost"] for c in children}

x = {name: solver.BoolVar(f"x_{name}") for name in names}

# Cardinality constraints
solver.Add(sum(x[name] for name in names) <= inp["max_children"])
solver.Add(sum(x[name] for name in names) >= inp["min_children"])

# Must-take constraints
for name in inp["must_take"]:
    solver.Add(x[name] == 1)

# Incompatibility constraints
for a, b in inp["incompatible_pairs"]:
    solver.Add(x[a] + x[b] <= 1)

# Implication constraints
for rule in inp["implications"]:
    a = rule["if_taken"]
    b = rule["must_also_take"]
    solver.Add(x[a] <= x[b])

# Objective: minimize total cost
solver.Minimize(sum(costs[name] * x[name] for name in names))

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
    selected_children = [name for name in names if x[name].solution_value() > 0.5]
    total_cost = sum(costs[name] for name in selected_children)
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

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))