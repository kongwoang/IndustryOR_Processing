import json
from ortools.linear_solver import pywraplp

inp = {
    "production_points": [
        {"id": 1, "supply": 100},
        {"id": 2, "supply": 150}
    ],
    "demand_points": [
        {"id": 1, "demand": 80},
        {"id": 2, "demand": 120}
    ],
    "stations": [
        {"id": 1, "fixed_cost": 10, "capacity": 100},
        {"id": 2, "fixed_cost": 15, "capacity": 100}
    ],
    "cost_source_to_station": [
        [2, 3],
        [4, 1]
    ],
    "cost_station_to_demand": [
        [3, 2],
        [1, 4]
    ]
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is unavailable.")

m = len(inp["production_points"])
n = len(inp["demand_points"])
p = len(inp["stations"])

x = {}
for i in range(m):
    for k in range(p):
        for j in range(n):
            x[i, k, j] = solver.NumVar(0.0, solver.infinity(), f"x_{i}_{k}_{j}")

y = {}
for k in range(p):
    y[k] = solver.BoolVar(f"y_{k}")

for i in range(m):
    solver.Add(
        sum(x[i, k, j] for k in range(p) for j in range(n)) <= inp["production_points"][i]["supply"]
    )

for j in range(n):
    solver.Add(
        sum(x[i, k, j] for i in range(m) for k in range(p)) == inp["demand_points"][j]["demand"]
    )

for k in range(p):
    solver.Add(
        sum(x[i, k, j] for i in range(m) for j in range(n)) <= inp["stations"][k]["capacity"] * y[k]
    )

objective = solver.Objective()
for i in range(m):
    for k in range(p):
        for j in range(n):
            unit_cost = inp["cost_source_to_station"][i][k] + inp["cost_station_to_demand"][k][j]
            objective.SetCoefficient(x[i, k, j], unit_cost)
for k in range(p):
    objective.SetCoefficient(y[k], inp["stations"][k]["fixed_cost"])
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

def clean_number(v):
    if v is None:
        return None
    if abs(v - round(v)) <= 1e-9:
        return int(round(v))
    return v

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "open_stations": [],
        "shipments": []
    }

    for k in range(p):
        throughput = sum(x[i, k, j].solution_value() for i in range(m) for j in range(n))
        output["open_stations"].append({
            "station_id": inp["stations"][k]["id"],
            "used": int(round(y[k].solution_value())),
            "throughput": clean_number(throughput)
        })

    for i in range(m):
        for k in range(p):
            for j in range(n):
                qty = x[i, k, j].solution_value()
                if qty > 1e-9:
                    output["shipments"].append({
                        "from_production": inp["production_points"][i]["id"],
                        "via_station": inp["stations"][k]["id"],
                        "to_demand": inp["demand_points"][j]["id"],
                        "quantity": clean_number(qty)
                    })

    objective_value = clean_number(solver.Objective().Value())
else:
    output = {
        "open_stations": [],
        "shipments": []
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, separators=(",", ":")))