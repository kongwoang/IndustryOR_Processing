import json
import math
from ortools.linear_solver import pywraplp

inp = {
    "demand_units": 1800,
    "modes": ["truck", "van", "motorcycle", "electric_vehicle"],
    "capacities_per_trip": {
        "truck": 100,
        "van": 80,
        "motorcycle": 40,
        "electric_vehicle": 60
    },
    "pollution_per_trip": {
        "truck": 100,
        "van": 50,
        "motorcycle": 10,
        "electric_vehicle": 0
    },
    "min_truck_trips": 10,
    "max_total_pollution": 2000,
    "mutually_exclusive_modes": ["van", "electric_vehicle"]
}

def create_solver():
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("No suitable MIP solver available in OR-Tools.")
    return solver

def build_model(fixed_pollution=None):
    solver = create_solver()
    modes = inp["modes"]
    caps = inp["capacities_per_trip"]
    pollution = inp["pollution_per_trip"]

    ub = math.ceil(inp["demand_units"] / min(caps.values()))

    x = {m: solver.IntVar(0, ub, f"x_{m}") for m in modes}
    use_van = solver.IntVar(0, 1, "use_van")
    use_ev = solver.IntVar(0, 1, "use_electric_vehicle")

    solver.Add(use_van + use_ev <= 1)
    solver.Add(x["van"] <= ub * use_van)
    solver.Add(x["electric_vehicle"] <= ub * use_ev)

    total_delivered = solver.Sum(caps[m] * x[m] for m in modes)
    total_pollution = solver.Sum(pollution[m] * x[m] for m in modes)

    solver.Add(total_delivered >= inp["demand_units"])
    solver.Add(total_pollution <= inp["max_total_pollution"])
    solver.Add(x["truck"] >= inp["min_truck_trips"])

    if fixed_pollution is None:
        solver.Minimize(total_pollution)
    else:
        solver.Add(total_pollution <= fixed_pollution)
        solver.Minimize(solver.Sum(x[m] for m in modes))

    return solver, x, total_delivered, total_pollution

def status_name(code):
    mapping = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    return mapping.get(code, str(code))

solver1, x1, delivered1, pollution1 = build_model()
status1 = solver1.Solve()

output = {
    "truck_trips": None,
    "van_trips": None,
    "motorcycle_trips": None,
    "electric_vehicle_trips": None,
    "total_delivered_units": None,
    "total_pollution": None
}
objective_value = None
final_status = status_name(status1)

if status1 == pywraplp.Solver.OPTIMAL:
    optimal_pollution = int(round(pollution1.solution_value()))

    solver2, x2, delivered2, pollution2 = build_model(fixed_pollution=optimal_pollution)
    status2 = solver2.Solve()

    if status2 == pywraplp.Solver.OPTIMAL:
        chosen_x = x2
        chosen_delivered = delivered2
        chosen_pollution = pollution2
        final_status = status_name(status2)
    else:
        chosen_x = x1
        chosen_delivered = delivered1
        chosen_pollution = pollution1

    output = {
        "truck_trips": int(round(chosen_x["truck"].solution_value())),
        "van_trips": int(round(chosen_x["van"].solution_value())),
        "motorcycle_trips": int(round(chosen_x["motorcycle"].solution_value())),
        "electric_vehicle_trips": int(round(chosen_x["electric_vehicle"].solution_value())),
        "total_delivered_units": int(round(chosen_delivered.solution_value())),
        "total_pollution": int(round(chosen_pollution.solution_value()))
    }
    objective_value = int(round(pollution1.solution_value()))

print(json.dumps({
    "status": final_status,
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))