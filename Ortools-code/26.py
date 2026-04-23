# Source: :contentReference[oaicite:0]{index=0}
import json
from ortools.linear_solver import pywraplp

inp = {
    "time_periods": [
        {"start": "2:00", "end": "6:00", "required_salespeople": 10},
        {"start": "6:00", "end": "10:00", "required_salespeople": 15},
        {"start": "10:00", "end": "14:00", "required_salespeople": 25},
        {"start": "14:00", "end": "18:00", "required_salespeople": 20},
        {"start": "18:00", "end": "22:00", "required_salespeople": 18},
        {"start": "22:00", "end": "2:00", "required_salespeople": 12}
    ],
    "shift_starts": ["2:00", "6:00", "10:00", "14:00", "18:00", "22:00"],
    "shift_length_hours": 8,
    "period_length_hours": 4
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

time_periods = inp["time_periods"]
shift_starts = inp["shift_starts"]
n = len(time_periods)
periods_per_shift = inp["shift_length_hours"] // inp["period_length_hours"]

x = {
    t: solver.IntVar(0, solver.infinity(), f"x_{t.replace(':', '_')}")
    for t in shift_starts
}

for i, period in enumerate(time_periods):
    covering_indices = [((i - k) % n) for k in range(periods_per_shift)]
    solver.Add(
        sum(x[shift_starts[idx]] for idx in covering_indices) >= period["required_salespeople"]
    )

solver.Minimize(sum(x[t] for t in shift_starts))

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "total_salespeople": int(round(solver.Objective().Value())),
        "staff_starting_at": {
            t: int(round(x[t].solution_value()))
            for t in shift_starts
        }
    }
    objective_value = int(round(solver.Objective().Value()))
else:
    output = {
        "total_salespeople": None,
        "staff_starting_at": {
            t: None for t in shift_starts
        }
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))