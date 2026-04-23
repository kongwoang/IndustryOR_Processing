import json
from ortools.linear_solver import pywraplp

inp = {
    "truck_types": [
        {
            "name": "Type A",
            "refrigerated_capacity": 20,
            "non_refrigerated_capacity": 40,
            "cost_per_km": 30
        },
        {
            "name": "Type B",
            "refrigerated_capacity": 30,
            "non_refrigerated_capacity": 30,
            "cost_per_km": 40
        }
    ],
    "demand": {
        "refrigerated": 3000,
        "non_refrigerated": 4000
    }
}

solver = pywraplp.Solver.CreateSolver("SCIP")

type_a = solver.IntVar(0, solver.infinity(), "type_a")
type_b = solver.IntVar(0, solver.infinity(), "type_b")

A = inp["truck_types"][0]
B = inp["truck_types"][1]
demand = inp["demand"]

solver.Add(
    A["refrigerated_capacity"] * type_a + B["refrigerated_capacity"] * type_b
    >= demand["refrigerated"]
)
solver.Add(
    A["non_refrigerated_capacity"] * type_a + B["non_refrigerated_capacity"] * type_b
    >= demand["non_refrigerated"]
)

solver.Minimize(A["cost_per_km"] * type_a + B["cost_per_km"] * type_b)

status = solver.Solve()

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "type_a_trucks": int(round(type_a.solution_value())),
        "type_b_trucks": int(round(type_b.solution_value())),
        "total_cost_per_km": int(round(solver.Objective().Value()))
    }
    result = {
        "status": "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE",
        "objective_value": int(round(solver.Objective().Value())),
        "example_output": output
    }
else:
    output = {
        "type_a_trucks": 0,
        "type_b_trucks": 0,
        "total_cost_per_km": 0
    }
    result = {
        "status": "INFEASIBLE",
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))