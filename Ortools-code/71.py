import json
from ortools.linear_solver import pywraplp

inp = {
    "n": 2,
    "cost_matrix": [
        [50, 160],
        [180, 280]
    ]
}

n = inp["n"]
cost = inp["cost_matrix"]

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

# x[i, p] = 1 if factory i is assigned to location p
x = {}
for i in range(n):
    for p in range(n):
        x[i, p] = solver.BoolVar(f"x_{i}_{p}")

# Each factory is assigned to exactly one location
for i in range(n):
    solver.Add(sum(x[i, p] for p in range(n)) == 1)

# Each location receives exactly one factory
for p in range(n):
    solver.Add(sum(x[i, p] for i in range(n)) == 1)

objective = solver.Objective()
for i in range(n):
    for p in range(n):
        objective.SetCoefficient(x[i, p], cost[i][p])
objective.SetMinimization()

status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    status_str = "OPTIMAL"
elif status == pywraplp.Solver.FEASIBLE:
    status_str = "FEASIBLE"
else:
    status_str = "INFEASIBLE_OR_NOT_SOLVED"

assignments = []
total_cost = None

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    for i in range(n):
        assigned_location = None
        for p in range(n):
            if x[i, p].solution_value() > 0.5:
                assigned_location = p + 1
                break
        assignments.append({
            "factory": i + 1,
            "location": assigned_location
        })
    total_cost = float(solver.Objective().Value())

output = {
    "assignments": assignments,
    "total_cost": total_cost
}

print(json.dumps({
    "status": status_str,
    "objective_value": total_cost,
    "example_output": output
}, ensure_ascii=False))