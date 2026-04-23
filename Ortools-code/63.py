import json
from ortools.linear_solver import pywraplp

inp = {
    "total_produce_required": 1000,
    "transport_options": {
        "horse": {
            "capacity_per_trip": 55,
            "pollution_per_trip": 80,
            "min_trips": 8
        },
        "bicycle": {
            "capacity_per_trip": 30,
            "pollution_per_trip": 0
        },
        "handcart": {
            "capacity_per_trip": 40,
            "pollution_per_trip": 0
        }
    },
    "max_total_pollution": 1000,
    "choose_only_one_of": [
        "bicycle",
        "handcart"
    ]
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

M = 10**6

x_horse = solver.IntVar(0, solver.infinity(), "horse_trips")
x_bicycle = solver.IntVar(0, solver.infinity(), "bicycle_trips")
x_handcart = solver.IntVar(0, solver.infinity(), "handcart_trips")

use_bicycle = solver.BoolVar("use_bicycle")
use_handcart = solver.BoolVar("use_handcart")

horse = inp["transport_options"]["horse"]
bicycle = inp["transport_options"]["bicycle"]
handcart = inp["transport_options"]["handcart"]

# At most one of bicycle and handcart can be used.
solver.Add(use_bicycle + use_handcart <= 1)
solver.Add(x_bicycle <= M * use_bicycle)
solver.Add(x_handcart <= M * use_handcart)

# Minimum horse trips.
solver.Add(x_horse >= horse["min_trips"])

# Pollution limit.
solver.Add(
    horse["pollution_per_trip"] * x_horse
    + bicycle["pollution_per_trip"] * x_bicycle
    + handcart["pollution_per_trip"] * x_handcart
    <= inp["max_total_pollution"]
)

# Required transported produce.
solver.Add(
    horse["capacity_per_trip"] * x_horse
    + bicycle["capacity_per_trip"] * x_bicycle
    + handcart["capacity_per_trip"] * x_handcart
    >= inp["total_produce_required"]
)

# Minimize total pollution.
objective = solver.Objective()
objective.SetCoefficient(x_horse, horse["pollution_per_trip"])
objective.SetCoefficient(x_bicycle, bicycle["pollution_per_trip"])
objective.SetCoefficient(x_handcart, handcart["pollution_per_trip"])
objective.SetMinimization()

status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    horse_trips = int(round(x_horse.solution_value()))
    bicycle_trips = int(round(x_bicycle.solution_value()))
    handcart_trips = int(round(x_handcart.solution_value()))
    total_transported = (
        horse["capacity_per_trip"] * horse_trips
        + bicycle["capacity_per_trip"] * bicycle_trips
        + handcart["capacity_per_trip"] * handcart_trips
    )
    total_pollution = (
        horse["pollution_per_trip"] * horse_trips
        + bicycle["pollution_per_trip"] * bicycle_trips
        + handcart["pollution_per_trip"] * handcart_trips
    )

    output = {
        "horse_trips": horse_trips,
        "bicycle_trips": bicycle_trips,
        "handcart_trips": handcart_trips,
        "total_transported": total_transported,
        "total_pollution": total_pollution
    }

    result = {
        "status": "OPTIMAL",
        "objective_value": solver.Objective().Value(),
        "example_output": output
    }
else:
    output = {
        "horse_trips": None,
        "bicycle_trips": None,
        "handcart_trips": None,
        "total_transported": None,
        "total_pollution": None
    }
    result = {
        "status": "INFEASIBLE_OR_NO_SOLUTION",
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result))