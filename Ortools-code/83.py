import json
from ortools.linear_solver import pywraplp

inp = {
    "period_length_hours": 4,
    "shift_length_hours": 8,
    "shift_start_times": [2, 6, 10, 14, 18, 22],
    "time_slots": ["2-6", "6-10", "10-14", "14-18", "18-22", "22-2"],
    "minimum_waiters_needed": [4, 8, 10, 7, 12, 4]
}


def build_model(fix_total=None, fixed_values=None, objective_index=None):
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP solver is unavailable.")

    n = len(inp["shift_start_times"])
    cover_span = inp["shift_length_hours"] // inp["period_length_hours"]

    x = [solver.IntVar(0, solver.infinity(), f"x_{inp['shift_start_times'][i]}") for i in range(n)]

    for i in range(n):
        solver.Add(
            solver.Sum(x[(i - k) % n] for k in range(cover_span)) >= inp["minimum_waiters_needed"][i]
        )

    if fix_total is None:
        solver.Minimize(solver.Sum(x))
    else:
        solver.Add(solver.Sum(x) == fix_total)
        if fixed_values:
            for idx, val in fixed_values.items():
                solver.Add(x[idx] == val)
        solver.Minimize(x[objective_index])

    return solver, x


# Phase 1: minimize total number of waiters
solver, x = build_model()
status = solver.Solve()

if status != pywraplp.Solver.OPTIMAL:
    output = {
        "total_waiters": None,
        "waiters_starting_by_time": {
            "2": None,
            "6": None,
            "10": None,
            "14": None,
            "18": None,
            "22": None
        }
    }
    result = {
        "status": "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "NOT_SOLVED",
        "objective_value": None,
        "example_output": output
    }
    print(json.dumps(result))
else:
    optimal_total = int(round(solver.Objective().Value()))

    # Phase 2: deterministic tie-breaking to get one canonical optimal schedule
    fixed = {}
    n = len(inp["shift_start_times"])
    for idx in range(n):
        tie_solver, tie_x = build_model(fix_total=optimal_total, fixed_values=fixed, objective_index=idx)
        tie_status = tie_solver.Solve()
        if tie_status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError("Tie-breaking model failed to solve optimally.")
        fixed[idx] = int(round(tie_x[idx].solution_value()))

    schedule = {
        str(inp["shift_start_times"][i]): fixed[i]
        for i in range(n)
    }

    output = {
        "total_waiters": optimal_total,
        "waiters_starting_by_time": schedule
    }

    result = {
        "status": "OPTIMAL",
        "objective_value": optimal_total,
        "example_output": output
    }

    print(json.dumps(result))