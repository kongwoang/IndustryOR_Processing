import json
from ortools.linear_solver import pywraplp

inp = {
    "time_periods": [
        {
            "shift": 1,
            "time": "6:00-10:00",
            "required_number": 60
        },
        {
            "shift": 2,
            "time": "10:00-14:00",
            "required_number": 70
        },
        {
            "shift": 3,
            "time": "14:00-18:00",
            "required_number": 60
        },
        {
            "shift": 4,
            "time": "18:00-22:00",
            "required_number": 50
        },
        {
            "shift": 5,
            "time": "22:00-2:00",
            "required_number": 20
        },
        {
            "shift": 6,
            "time": "2:00-6:00",
            "required_number": 30
        }
    ],
    "work_hours_per_staff": 8,
    "hours_per_time_period": 4
}

periods = inp["time_periods"]
n = len(periods)
cover_len = inp["work_hours_per_staff"] // inp["hours_per_time_period"]

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

x = []
for i in range(n):
    x.append(solver.IntVar(0, solver.infinity(), f"x_{i+1}"))

for i in range(n):
    covered_by = []
    for k in range(cover_len):
        covered_by.append(x[(i - k) % n])
    solver.Add(solver.Sum(covered_by) >= periods[i]["required_number"])

solver.Minimize(solver.Sum(x))

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
    output = {
        "minimum_total_staff": int(round(solver.Objective().Value())),
        "staff_starting_each_shift": [
            {
                "shift": periods[i]["shift"],
                "time": periods[i]["time"],
                "count": int(round(x[i].solution_value()))
            }
            for i in range(n)
        ]
    }
    objective_value = int(round(solver.Objective().Value()))
else:
    output = {
        "minimum_total_staff": None,
        "staff_starting_each_shift": []
    }
    objective_value = None

result = {
    "status": status_map.get(status_code, str(status_code)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))