import json
from ortools.linear_solver import pywraplp

inp = {
    "methods": [
        {
            "name": "motorcycle",
            "pollution_per_trip": 40,
            "capacity_per_trip": 10,
            "max_trips": 8
        },
        {
            "name": "small_truck",
            "pollution_per_trip": 70,
            "capacity_per_trip": 20,
            "max_trips": None
        },
        {
            "name": "large_truck",
            "pollution_per_trip": 100,
            "capacity_per_trip": 50,
            "max_trips": None
        }
    ],
    "max_selected_methods": 2,
    "min_total_capacity": 300,
    "max_total_trips": 20,
    "objective": "minimize_total_pollution"
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

methods = inp["methods"]
method_names = [m["name"] for m in methods]

x = {}
y = {}

for m in methods:
    name = m["name"]
    ub = m["max_trips"] if m["max_trips"] is not None else inp["max_total_trips"]
    x[name] = solver.IntVar(0, ub, f"x_{name}")
    y[name] = solver.BoolVar(f"y_{name}")

    solver.Add(x[name] <= ub * y[name])

    if m["max_trips"] is not None:
        solver.Add(x[name] <= m["max_trips"])

solver.Add(solver.Sum(y[name] for name in method_names) <= inp["max_selected_methods"])
solver.Add(solver.Sum(x[name] for name in method_names) <= inp["max_total_trips"])
solver.Add(
    solver.Sum(x[m["name"]] * m["capacity_per_trip"] for m in methods) >= inp["min_total_capacity"]
)

solver.Minimize(
    solver.Sum(x[m["name"]] * m["pollution_per_trip"] for m in methods)
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
status_str = status_map.get(status, str(status))

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    trips = {name: int(round(x[name].solution_value())) for name in method_names}
    total_trips = sum(trips.values())
    total_capacity = sum(
        trips[m["name"]] * m["capacity_per_trip"] for m in methods
    )
    total_pollution = sum(
        trips[m["name"]] * m["pollution_per_trip"] for m in methods
    )
    selected_methods = [name for name in method_names if trips[name] > 0]

    output = {
        "selected_methods": selected_methods,
        "trips": {
            "motorcycle": trips["motorcycle"],
            "small_truck": trips["small_truck"],
            "large_truck": trips["large_truck"]
        },
        "total_trips": total_trips,
        "total_capacity": total_capacity,
        "total_pollution": total_pollution
    }
    objective_value = float(solver.Objective().Value())
else:
    output = {
        "selected_methods": [],
        "trips": {
            "motorcycle": 0,
            "small_truck": 0,
            "large_truck": 0
        },
        "total_trips": 0,
        "total_capacity": 0,
        "total_pollution": 0
    }
    objective_value = None

print(json.dumps({
    "status": status_str,
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))