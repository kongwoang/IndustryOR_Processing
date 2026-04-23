import json
from ortools.linear_solver import pywraplp

inp = {
    "workers": ["I", "II", "III", "IV", "V"],
    "tasks": ["A", "B", "C", "D"],
    "hours": {
        "I": {"A": 9, "B": 4, "C": 3, "D": 7},
        "II": {"A": 4, "B": 6, "C": 5, "D": 6},
        "III": {"A": 5, "B": 4, "C": 7, "D": 5},
        "IV": {"A": 7, "B": 5, "C": 2, "D": 3},
        "V": {"A": 10, "B": 6, "C": 7, "D": 4}
    }
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

workers = inp["workers"]
tasks = inp["tasks"]
hours = inp["hours"]

x = {}
for w in workers:
    for t in tasks:
        x[(w, t)] = solver.BoolVar(f"x_{w}_{t}")

for t in tasks:
    solver.Add(sum(x[(w, t)] for w in workers) == 1)

for w in workers:
    solver.Add(sum(x[(w, t)] for t in tasks) <= 1)

solver.Minimize(
    sum(hours[w][t] * x[(w, t)] for w in workers for t in tasks)
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

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    assignments = []
    assigned_workers = set()
    for t in tasks:
        for w in workers:
            if x[(w, t)].solution_value() > 0.5:
                assignments.append({
                    "worker": w,
                    "task": t,
                    "hours": hours[w][t]
                })
                assigned_workers.add(w)

    unassigned_worker = next((w for w in workers if w not in assigned_workers), None)
    total_hours = int(round(solver.Objective().Value()))

    output = {
        "total_hours": total_hours,
        "assignments": assignments,
        "unassigned_worker": unassigned_worker
    }

    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": total_hours,
        "example_output": output
    }
else:
    output = {
        "total_hours": None,
        "assignments": [],
        "unassigned_worker": None
    }
    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))