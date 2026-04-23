import json
import math
from ortools.linear_solver import pywraplp

inp = {
    "depot": {
        "id": 0,
        "x": 40,
        "y": 50,
        "time_window": [0, 1236]
    },
    "customers": [
        {"id": 1, "x": 45, "y": 68, "demand": 10, "time_window": [912, 967], "service_duration": 90},
        {"id": 2, "x": 45, "y": 70, "demand": 30, "time_window": [825, 870], "service_duration": 90},
        {"id": 3, "x": 42, "y": 66, "demand": 10, "time_window": [65, 146], "service_duration": 90},
        {"id": 4, "x": 42, "y": 68, "demand": 10, "time_window": [727, 782], "service_duration": 90},
        {"id": 5, "x": 42, "y": 65, "demand": 10, "time_window": [15, 67], "service_duration": 90},
        {"id": 6, "x": 40, "y": 69, "demand": 20, "time_window": [621, 702], "service_duration": 90},
        {"id": 7, "x": 40, "y": 66, "demand": 20, "time_window": [170, 225], "service_duration": 90},
        {"id": 8, "x": 38, "y": 68, "demand": 20, "time_window": [255, 324], "service_duration": 90},
        {"id": 9, "x": 38, "y": 70, "demand": 10, "time_window": [534, 605], "service_duration": 90},
        {"id": 10, "x": 35, "y": 66, "demand": 10, "time_window": [357, 410], "service_duration": 90},
        {"id": 11, "x": 35, "y": 69, "demand": 10, "time_window": [448, 505], "service_duration": 90},
        {"id": 12, "x": 25, "y": 85, "demand": 20, "time_window": [652, 721], "service_duration": 90},
        {"id": 13, "x": 22, "y": 75, "demand": 30, "time_window": [30, 92], "service_duration": 90},
        {"id": 14, "x": 22, "y": 85, "demand": 10, "time_window": [567, 620], "service_duration": 90},
        {"id": 15, "x": 20, "y": 80, "demand": 40, "time_window": [384, 429], "service_duration": 90},
        {"id": 16, "x": 20, "y": 85, "demand": 40, "time_window": [475, 528], "service_duration": 90},
        {"id": 17, "x": 18, "y": 75, "demand": 20, "time_window": [99, 148], "service_duration": 90},
        {"id": 18, "x": 15, "y": 75, "demand": 20, "time_window": [179, 254], "service_duration": 90},
        {"id": 19, "x": 15, "y": 80, "demand": 10, "time_window": [278, 345], "service_duration": 90},
        {"id": 20, "x": 30, "y": 50, "demand": 10, "time_window": [10, 73], "service_duration": 90}
    ],
    "vehicle_count_max": 5,
    "vehicle_capacity": 200,
    "objective": "minimize_total_distance",
    "distance_metric": "euclidean",
    "travel_time_equals_distance": True
}

output_format = {
    "routes": [
        {
            "vehicle_id": "integer",
            "stops": ["integer"],
            "load": "number",
            "distance": "number",
            "arrival_times": ["number"],
            "service_start_times": ["number"]
        }
    ],
    "total_distance": "number"
}

depot = inp["depot"]
customers = inp["customers"]
K = inp["vehicle_count_max"]
Q = inp["vehicle_capacity"]

nodes = [depot] + customers
customer_ids = [c["id"] for c in customers]
all_ids = [node["id"] for node in nodes]

x_coord = {node["id"]: node["x"] for node in nodes}
y_coord = {node["id"]: node["y"] for node in nodes}
demand = {depot["id"]: 0}
service = {depot["id"]: 0}
tw_start = {depot["id"]: depot["time_window"][0]}
tw_end = {depot["id"]: depot["time_window"][1]}

for c in customers:
    demand[c["id"]] = c["demand"]
    service[c["id"]] = c["service_duration"]
    tw_start[c["id"]] = c["time_window"][0]
    tw_end[c["id"]] = c["time_window"][1]

dist = {}
for i in all_ids:
    dist[i] = {}
    for j in all_ids:
        if i == j:
            dist[i][j] = 0.0
        else:
            dist[i][j] = math.hypot(x_coord[i] - x_coord[j], y_coord[i] - y_coord[j])

travel_time = dist

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

x = {}
for k in range(K):
    for i in all_ids:
        for j in all_ids:
            if i != j:
                x[i, j, k] = solver.BoolVar(f"x_{i}_{j}_{k}")

use = {k: solver.BoolVar(f"use_{k}") for k in range(K)}

s = {}
for k in range(K):
    for i in customer_ids:
        s[i, k] = solver.NumVar(0.0, solver.infinity(), f"s_{i}_{k}")

a = {}
for k in range(K):
    for i in customer_ids:
        a[i, k] = solver.NumVar(0.0, solver.infinity(), f"a_{i}_{k}")

u = {}
for k in range(K):
    for i in customer_ids:
        u[i, k] = solver.NumVar(0.0, len(customer_ids), f"u_{i}_{k}")

for i in customer_ids:
    solver.Add(sum(x[j, i, k] for k in range(K) for j in all_ids if j != i) == 1)
    solver.Add(sum(x[i, j, k] for k in range(K) for j in all_ids if j != i) == 1)

for k in range(K):
    solver.Add(sum(x[0, j, k] for j in customer_ids) == use[k])
    solver.Add(sum(x[i, 0, k] for i in customer_ids) == use[k])
    for h in customer_ids:
        solver.Add(
            sum(x[i, h, k] for i in all_ids if i != h) ==
            sum(x[h, j, k] for j in all_ids if j != h)
        )

for k in range(K):
    solver.Add(
        sum(demand[i] * sum(x[j, i, k] for j in all_ids if j != i) for i in customer_ids) <= Q
    )

M_time = depot["time_window"][1] + max(service.values()) + max(
    max(travel_time[i].values()) for i in all_ids
)

for k in range(K):
    for i in customer_ids:
        visit_ik = sum(x[j, i, k] for j in all_ids if j != i)
        solver.Add(a[i, k] >= sum(travel_time[0][i] * x[0, i, k] for _ in [0]))
        solver.Add(a[i, k] <= M_time * visit_ik)
        solver.Add(s[i, k] >= a[i, k])
        solver.Add(s[i, k] >= tw_start[i] * visit_ik)
        solver.Add(s[i, k] <= tw_end[i] + M_time * (1 - visit_ik))
        solver.Add(s[i, k] <= M_time * visit_ik)
        solver.Add(u[i, k] >= visit_ik)
        solver.Add(u[i, k] <= len(customer_ids) * visit_ik)

for k in range(K):
    for i in customer_ids:
        for j in customer_ids:
            if i != j:
                solver.Add(
                    s[j, k] >= s[i, k] + service[i] + travel_time[i][j] - M_time * (1 - x[i, j, k])
                )
                solver.Add(
                    a[j, k] >= s[i, k] + service[i] + travel_time[i][j] - M_time * (1 - x[i, j, k])
                )

for k in range(K):
    for j in customer_ids:
        solver.Add(
            a[j, k] >= travel_time[0][j] - M_time * (1 - x[0, j, k])
        )

for k in range(K):
    for i in customer_ids:
        solver.Add(
            s[i, k] + service[i] + travel_time[i][0] <= depot["time_window"][1] + M_time * (1 - x[i, 0, k])
        )

Ncust = len(customer_ids)
for k in range(K):
    for i in customer_ids:
        for j in customer_ids:
            if i != j:
                solver.Add(
                    u[i, k] - u[j, k] + Ncust * x[i, j, k] <= Ncust - 1
                )

objective = solver.Objective()
for k in range(K):
    for i in all_ids:
        for j in all_ids:
            if i != j:
                objective.SetCoefficient(x[i, j, k], dist[i][j])
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

routes = []
total_distance = None

if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    total_distance = objective.Value()

    for k in range(K):
        if use[k].solution_value() < 0.5:
            continue

        current = 0
        route_stops = [0]
        arrival_times = [0.0]
        service_start_times = [0.0]
        load = 0.0
        route_distance = 0.0

        while True:
            next_node = None
            for j in all_ids:
                if j != current and (current, j, k) in x and x[current, j, k].solution_value() > 0.5:
                    next_node = j
                    break

            if next_node is None:
                break

            route_distance += dist[current][next_node]

            if next_node == 0:
                route_stops.append(0)
                arrival_to_depot = service_start_times[-1]
                prev_node = route_stops[-2]
                if prev_node != 0:
                    arrival_to_depot = service_start_times[-1] + service[prev_node] + travel_time[prev_node][0]
                arrival_times.append(round(arrival_to_depot, 6))
                service_start_times.append(round(arrival_to_depot, 6))
                break
            else:
                route_stops.append(next_node)
                load += demand[next_node]
                arrival_times.append(round(a[next_node, k].solution_value(), 6))
                service_start_times.append(round(s[next_node, k].solution_value(), 6))
                current = next_node

        routes.append({
            "vehicle_id": k,
            "stops": route_stops,
            "load": round(load, 6),
            "distance": round(route_distance, 6),
            "arrival_times": arrival_times,
            "service_start_times": service_start_times
        })

    output = {
        "routes": routes,
        "total_distance": round(total_distance, 6)
    }
else:
    output = {
        "routes": [],
        "total_distance": None
    }

print(json.dumps({
    "status": status_map.get(status_code, str(status_code)),
    "objective_value": None if total_distance is None else round(total_distance, 2),
    "example_output": output
}, ensure_ascii=False))