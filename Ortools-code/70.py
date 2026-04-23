import json
from ortools.linear_solver import pywraplp

inp = {
    "cities": [1, 2, 3, 4],
    "distance_matrix": [
        [0, 10, 20, 12],
        [10, 0, 5, 10],
        [20, 5, 0, 8],
        [15, 12, 8, 0]
    ],
    "start_city": 1
}

cities = inp["cities"]
distance_matrix = inp["distance_matrix"]
start_city = inp["start_city"]
n = len(cities)

idx = {city: k for k, city in enumerate(cities)}
dist = {
    (i, j): distance_matrix[idx[i]][idx[j]]
    for i in cities for j in cities
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

# Decision variables: x[i,j] = 1 if the route goes directly from city i to city j
x = {}
for i in cities:
    for j in cities:
        if i != j:
            x[i, j] = solver.BoolVar(f"x_{i}_{j}")

# MTZ ordering variables to eliminate subtours
u = {}
for i in cities:
    if i != start_city:
        u[i] = solver.IntVar(1, n - 1, f"u_{i}")

# Each city has exactly one outgoing arc
for i in cities:
    solver.Add(sum(x[i, j] for j in cities if j != i) == 1)

# Each city has exactly one incoming arc
for j in cities:
    solver.Add(sum(x[i, j] for i in cities if i != j) == 1)

# MTZ subtour elimination constraints
for i in cities:
    if i == start_city:
        continue
    for j in cities:
        if j == start_city or i == j:
            continue
        solver.Add(u[i] - u[j] + n * x[i, j] <= n - 1)

# Objective: minimize total travel distance
solver.Minimize(
    sum(dist[i, j] * x[i, j] for i in cities for j in cities if i != j)
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

output = {
    "route": [],
    "total_distance": None
}

objective_value = None

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    successor = {}
    for i in cities:
        for j in cities:
            if i != j and x[i, j].solution_value() > 0.5:
                successor[i] = j

    route = [start_city]
    current = start_city
    while True:
        nxt = successor[current]
        route.append(nxt)
        if nxt == start_city:
            break
        current = nxt

    objective_value = solver.Objective().Value()
    output = {
        "route": route,
        "total_distance": objective_value
    }

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))