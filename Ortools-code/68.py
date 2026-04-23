import json
from ortools.linear_solver import pywraplp

inp = {
    "sources": [
        {"name": "A", "supply_tons": 80},
        {"name": "B", "supply_tons": 100}
    ],
    "destinations": [
        {"name": "ResidentialArea1", "demand_tons": 55},
        {"name": "ResidentialArea2", "demand_tons": 75},
        {"name": "ResidentialArea3", "demand_tons": 50}
    ],
    "distance_km": [
        [10, 5, 6],
        [4, 8, 15]
    ]
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create solver.")

sources = inp["sources"]
destinations = inp["destinations"]
distance = inp["distance_km"]

m = len(sources)
n = len(destinations)

x = {}
for i in range(m):
    for j in range(n):
        x[i, j] = solver.NumVar(0.0, solver.infinity(), f"x_{i}_{j}")

for i in range(m):
    solver.Add(sum(x[i, j] for j in range(n)) == sources[i]["supply_tons"])

for j in range(n):
    solver.Add(sum(x[i, j] for i in range(m)) == destinations[j]["demand_tons"])

objective = solver.Objective()
for i in range(m):
    for j in range(n):
        objective.SetCoefficient(x[i, j], distance[i][j])
objective.SetMinimization()

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
    "shipments_tons": []
}

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    for i in range(m):
        for j in range(n):
            val = x[i, j].solution_value()
            output["shipments_tons"].append({
                "source": sources[i]["name"],
                "destination": destinations[j]["name"],
                "tons": val
            })
    objective_value = objective.Value()
else:
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))