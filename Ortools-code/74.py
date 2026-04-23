# Source: :contentReference[oaicite:0]{index=0}
import json
from ortools.linear_solver import pywraplp

inp = {
    "activities": [
        {
            "name": "A",
            "duration": 4
        },
        {
            "name": "B",
            "duration": 3
        },
        {
            "name": "C",
            "duration": 5
        },
        {
            "name": "D",
            "duration": 2
        },
        {
            "name": "E",
            "duration": 10
        },
        {
            "name": "F",
            "duration": 10
        },
        {
            "name": "G",
            "duration": 1
        }
    ],
    "precedence": [
        ["A", "G"],
        ["A", "D"],
        ["E", "F"],
        ["G", "F"],
        ["D", "C"],
        ["F", "C"],
        ["F", "B"]
    ],
    "daily_work_cost": 1000,
    "machine_rental": {
        "from_activity_start": "A",
        "to_activity_end": "B",
        "daily_cost": 5000
    }
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("GLOP solver is not available.")

activities = [a["name"] for a in inp["activities"]]
dur = {a["name"]: a["duration"] for a in inp["activities"]}

start = {a: solver.NumVar(0.0, solver.infinity(), f"start_{a}") for a in activities}
project_completion_time = solver.NumVar(0.0, solver.infinity(), "project_completion_time")
machine_rental_days = solver.NumVar(0.0, solver.infinity(), "machine_rental_days")

for pred, succ in inp["precedence"]:
    solver.Add(start[succ] >= start[pred] + dur[pred])

for a in activities:
    solver.Add(project_completion_time >= start[a] + dur[a])

machine_start_activity = inp["machine_rental"]["from_activity_start"]
machine_end_activity = inp["machine_rental"]["to_activity_end"]
solver.Add(machine_rental_days >= start[machine_end_activity] + dur[machine_end_activity] - start[machine_start_activity])

solver.Minimize(
    inp["daily_work_cost"] * project_completion_time
    + inp["machine_rental"]["daily_cost"] * machine_rental_days
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

def r(x):
    return round(float(x), 6)

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    start_times = {a: r(start[a].solution_value()) for a in activities}
    finish_times = {a: r(start[a].solution_value() + dur[a]) for a in activities}
    pct = r(project_completion_time.solution_value())
    mrd = r(machine_rental_days.solution_value())
    work_cost = r(inp["daily_work_cost"] * project_completion_time.solution_value())
    machine_cost = r(inp["machine_rental"]["daily_cost"] * machine_rental_days.solution_value())

    output = {
        "start_times": start_times,
        "finish_times": finish_times,
        "project_completion_time": pct,
        "machine_rental_days": mrd,
        "cost_breakdown": {
            "work_cost": work_cost,
            "machine_cost": machine_cost
        }
    }

    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": r(solver.Objective().Value()),
        "example_output": output
    }
else:
    output = {
        "start_times": {},
        "finish_times": {},
        "project_completion_time": None,
        "machine_rental_days": None,
        "cost_breakdown": {
            "work_cost": None,
            "machine_cost": None
        }
    }
    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))