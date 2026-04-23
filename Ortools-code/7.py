import json
import math
from ortools.linear_solver import pywraplp

inp = {
    "warehouses": [
        "Verona",
        "Perugia",
        "Rome",
        "Pescara",
        "Taranto",
        "Lamezia"
    ],
    "ports": [
        "Genoa",
        "Venice",
        "Ancona",
        "Naples",
        "Bari"
    ],
    "supply": {
        "Verona": 10,
        "Perugia": 12,
        "Rome": 20,
        "Pescara": 24,
        "Taranto": 18,
        "Lamezia": 40
    },
    "demand": {
        "Genoa": 20,
        "Venice": 15,
        "Ancona": 25,
        "Naples": 33,
        "Bari": 21
    },
    "distance_km": {
        "Verona": {
            "Genoa": 290,
            "Venice": 115,
            "Ancona": 355,
            "Naples": 715,
            "Bari": 810
        },
        "Perugia": {
            "Genoa": 380,
            "Venice": 340,
            "Ancona": 165,
            "Naples": 380,
            "Bari": 610
        },
        "Rome": {
            "Genoa": 505,
            "Venice": 530,
            "Ancona": 285,
            "Naples": 220,
            "Bari": 450
        },
        "Pescara": {
            "Genoa": 655,
            "Venice": 450,
            "Ancona": 155,
            "Naples": 240,
            "Bari": 315
        },
        "Taranto": {
            "Genoa": 1010,
            "Venice": 840,
            "Ancona": 550,
            "Naples": 305,
            "Bari": 95
        },
        "Lamezia": {
            "Genoa": 1072,
            "Venice": 1097,
            "Ancona": 747,
            "Naples": 372,
            "Bari": 333
        }
    },
    "truck_capacity_containers": 2,
    "container_cost_eur_per_km": 30
}

warehouses = inp["warehouses"]
ports = inp["ports"]
supply = inp["supply"]
demand = inp["demand"]
distance_km = inp["distance_km"]
truck_capacity = inp["truck_capacity_containers"]
container_cost_eur_per_km = inp["container_cost_eur_per_km"]

solver = pywraplp.Solver.CreateSolver("GLOP")

def zero_output():
    return {
        "shipments_containers": {
            i: {j: 0 for j in ports} for i in warehouses
        },
        "shipments_trucks": {
            i: {j: 0 for j in ports} for i in warehouses
        },
        "unused_containers": {
            i: supply[i] for i in warehouses
        }
    }

if solver is None:
    print(json.dumps({
        "status": "SOLVER_NOT_AVAILABLE",
        "objective_value": None,
        "example_output": zero_output()
    }, ensure_ascii=False))
    raise SystemExit

x = {}
for i in warehouses:
    for j in ports:
        x[i, j] = solver.NumVar(0.0, solver.infinity(), f"x_{i}_{j}")

for i in warehouses:
    solver.Add(sum(x[i, j] for j in ports) <= supply[i])

for j in ports:
    solver.Add(sum(x[i, j] for i in warehouses) == demand[j])

solver.Minimize(
    sum(
        container_cost_eur_per_km * distance_km[i][j] * x[i, j]
        for i in warehouses
        for j in ports
    )
)

status_code = solver.Solve()
status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}
status = status_map.get(status_code, f"UNKNOWN_STATUS_{status_code}")

output = zero_output()
objective_value = None

if status in {"OPTIMAL", "FEASIBLE"}:
    for i in warehouses:
        shipped_total = 0
        for j in ports:
            val = x[i, j].solution_value()
            containers = int(round(val))
            if abs(val - containers) > 1e-6:
                containers = int(math.floor(val + 0.5))
            trucks = math.ceil(containers / truck_capacity) if containers > 0 else 0
            output["shipments_containers"][i][j] = containers
            output["shipments_trucks"][i][j] = trucks
            shipped_total += containers
        output["unused_containers"][i] = supply[i] - shipped_total
    objective_value = float(round(solver.Objective().Value(), 6))

print(json.dumps({
    "status": status,
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))