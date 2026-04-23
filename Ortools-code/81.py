import json
from ortools.linear_solver import pywraplp

inp = {
    "car_lengths_meters": [4, 4.5, 5, 4.1, 2.4, 5.2, 3.7, 3.5, 3.2, 4.5, 2.3, 3.3, 3.8, 4.6, 3],
    "capped_side_limit_meters": 30
}

def solve_instance(car_lengths_meters, capped_side_limit_meters=None):
    scale = 10
    lengths = [int(round(v * scale)) for v in car_lengths_meters]
    n = len(lengths)
    total = sum(lengths)

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP solver is not available.")

    x = [solver.BoolVar(f"x[{i}]") for i in range(n)]
    z = solver.NumVar(0.0, solver.infinity(), "z")

    side_1 = solver.Sum(lengths[i] * x[i] for i in range(n))
    side_2 = total - side_1

    solver.Add(side_1 <= z)
    solver.Add(side_2 <= z)

    if capped_side_limit_meters is not None:
        solver.Add(side_1 <= int(round(capped_side_limit_meters * scale)))

    solver.Minimize(z)

    status_code = solver.Solve()
    status_map = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    status = status_map.get(status_code, str(status_code))

    if status_code not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        result = {
            "status": status,
            "occupied_street_length_meters": None,
            "side_1_cars": [],
            "side_1_total_meters": None,
            "side_2_cars": [],
            "side_2_total_meters": None
        }
        if capped_side_limit_meters is not None:
            result["side_1_cap_meters"] = capped_side_limit_meters
        return result

    side_1_cars = [i + 1 for i in range(n) if x[i].solution_value() > 0.5]
    side_2_cars = [i + 1 for i in range(n) if x[i].solution_value() <= 0.5]

    side_1_total = sum(lengths[i - 1] for i in side_1_cars)
    side_2_total = sum(lengths[i - 1] for i in side_2_cars)

    result = {
        "status": status,
        "occupied_street_length_meters": round(z.solution_value() / scale, 1),
        "side_1_cars": side_1_cars,
        "side_1_total_meters": round(side_1_total / scale, 1),
        "side_2_cars": side_2_cars,
        "side_2_total_meters": round(side_2_total / scale, 1)
    }
    if capped_side_limit_meters is not None:
        result["side_1_cap_meters"] = capped_side_limit_meters
    return result

base_case = solve_instance(inp["car_lengths_meters"])
capped_side_case = solve_instance(inp["car_lengths_meters"], inp["capped_side_limit_meters"])

output = {
    "base_case": base_case,
    "capped_side_case": capped_side_case
}

top_status = "OPTIMAL" if (
    base_case["status"] == "OPTIMAL" and capped_side_case["status"] == "OPTIMAL"
) else "NON_OPTIMAL"

result = {
    "status": top_status,
    "objective_value": output["base_case"]["occupied_street_length_meters"],
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))