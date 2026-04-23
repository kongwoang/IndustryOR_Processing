import json
from ortools.linear_solver import pywraplp

inp = {
    "students": 400,
    "drivers_available": 9,
    "vehicles": {
        "bus": {
            "available": 10,
            "seats": 50,
            "rental_cost": 800
        },
        "minibus": {
            "available": 8,
            "seats": 40,
            "rental_cost": 600
        }
    }
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

x_bus = solver.IntVar(0, inp["vehicles"]["bus"]["available"], "buses_used")
x_minibus = solver.IntVar(0, inp["vehicles"]["minibus"]["available"], "minibuses_used")

solver.Add(
    inp["vehicles"]["bus"]["seats"] * x_bus +
    inp["vehicles"]["minibus"]["seats"] * x_minibus
    >= inp["students"]
)

solver.Add(x_bus + x_minibus <= inp["drivers_available"])

solver.Minimize(
    inp["vehicles"]["bus"]["rental_cost"] * x_bus +
    inp["vehicles"]["minibus"]["rental_cost"] * x_minibus
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

if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    buses_used = int(round(x_bus.solution_value()))
    minibuses_used = int(round(x_minibus.solution_value()))
    total_seats = (
        inp["vehicles"]["bus"]["seats"] * buses_used +
        inp["vehicles"]["minibus"]["seats"] * minibuses_used
    )
    total_cost = (
        inp["vehicles"]["bus"]["rental_cost"] * buses_used +
        inp["vehicles"]["minibus"]["rental_cost"] * minibuses_used
    )
    output = {
        "buses_used": buses_used,
        "minibuses_used": minibuses_used,
        "total_seats": total_seats,
        "drivers_used": buses_used + minibuses_used,
        "total_cost": total_cost
    }
    objective_value = solver.Objective().Value()
else:
    output = {
        "buses_used": 0,
        "minibuses_used": 0,
        "total_seats": 0,
        "drivers_used": 0,
        "total_cost": 0
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status_code, str(status_code)),
    "objective_value": objective_value,
    "example_output": output
}, separators=(",", ":")))