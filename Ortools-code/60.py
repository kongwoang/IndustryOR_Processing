import json
from ortools.linear_solver import pywraplp

inp = {
    "num_locations": 7,
    "start_location": 1,
    "distance_matrix": [
        [0, 86, 49, 57, 31, 69, 50],
        [86, 0, 68, 79, 93, 24, 5],
        [49, 68, 0, 16, 7, 72, 67],
        [57, 79, 16, 0, 90, 69, 1],
        [31, 93, 7, 90, 0, 86, 59],
        [69, 24, 72, 69, 86, 0, 81],
        [50, 5, 67, 1, 59, 81, 0]
    ]
}

n = inp["num_locations"]
start = inp["start_location"] - 1
dist = inp["distance_matrix"]

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

# Directed TSP with MTZ subtour elimination
x = {}
for i in range(n):
    for j in range(n):
        if i != j:
            x[i, j] = solver.BoolVar(f"x_{i}_{j}")

u = {}
for i in range(n):
    if i != start:
        u[i] = solver.NumVar(1, n - 1, f"u_{i}")

# Each node has exactly one outgoing arc
for i in range(n):
    solver.Add(sum(x[i, j] for j in range(n) if j != i) == 1)

# Each node has exactly one incoming arc
for j in range(n):
    solver.Add(sum(x[i, j] for i in range(n) if i != j) == 1)

# MTZ constraints for subtour elimination (excluding start node)
for i in range(n):
    if i == start:
        continue
    for j in range(n):
        if j == start or j == i:
            continue
        solver.Add(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2)

objective = solver.Objective()
for i in range(n):
    for j in range(n):
        if i != j:
            objective.SetCoefficient(x[i, j], dist[i][j])
objective.SetMinimization()

status_code = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    succ = {}
    for i in range(n):
        for j in range(n):
            if i != j and x[i, j].solution_value() > 0.5:
                succ[i] = j
                break

    tour = [start + 1]
    current = start
    for _ in range(n - 1):
        current = succ[current]
        tour.append(current + 1)
    tour.append(start + 1)

    total_distance = sum(dist[tour[k] - 1][tour[k + 1] - 1] for k in range(len(tour) - 1))

    output = {
        "tour": tour,
        "total_distance": total_distance
    }
    objective_value = solver.Objective().Value()
else:
    output = {
        "tour": [],
        "total_distance": None
    }
    objective_value = None

result = {
    "status": status_map.get(status_code, str(status_code)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))