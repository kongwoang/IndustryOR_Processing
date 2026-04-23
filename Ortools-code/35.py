import json
from ortools.linear_solver import pywraplp

inp = {
    "batches": [1, 2, 3, 4, 5],
    "vats": [1, 2, 3],
    "processing_times": [
        [3.0, 1.0, 1.0],
        [2.0, 1.5, 1.0],
        [3.0, 1.2, 1.3],
        [2.0, 2.0, 2.0],
        [2.1, 2.0, 3.0]
    ]
}

batches = inp["batches"]
vats = inp["vats"]
p = inp["processing_times"]

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    solver = pywraplp.Solver.CreateSolver("CBC")
if solver is None:
    raise RuntimeError("No suitable MIP solver available in OR-Tools.")

n = len(batches)
m = len(vats)
horizon = sum(sum(row) for row in p)

start = {}
for i in range(n):
    for j in range(m):
        start[i, j] = solver.NumVar(0.0, horizon, f"start_{i}_{j}")

cmax = solver.NumVar(0.0, horizon, "makespan")

order = {}
for j in range(m):
    for i in range(n):
        for k in range(i + 1, n):
            order[i, k, j] = solver.BoolVar(f"order_{i}_{k}_{j}")

for i in range(n):
    for j in range(m - 1):
        solver.Add(start[i, j + 1] >= start[i, j] + p[i][j])

for j in range(m):
    for i in range(n):
        for k in range(i + 1, n):
            y = order[i, k, j]
            solver.Add(start[k, j] >= start[i, j] + p[i][j] - horizon * (1 - y))
            solver.Add(start[i, j] >= start[k, j] + p[k][j] - horizon * y)

for i in range(n):
    solver.Add(cmax >= start[i, m - 1] + p[i][m - 1])

solver.Minimize(cmax)

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}
status_str = status_map.get(status, "UNKNOWN")

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    vat_schedules = []
    for j in range(m):
        ops = []
        for i in range(n):
            st = start[i, j].solution_value()
            en = st + p[i][j]
            ops.append({
                "batch": batches[i],
                "start": round(st, 6),
                "end": round(en, 6)
            })
        ops.sort(key=lambda x: (x["start"], x["batch"]))
        vat_schedules.append({
            "vat": vats[j],
            "operations": ops
        })

    output = {
        "makespan": round(cmax.solution_value(), 6),
        "vat_schedules": vat_schedules
    }

    result = {
        "status": status_str,
        "objective_value": round(cmax.solution_value(), 6),
        "example_output": output
    }
else:
    output = {
        "makespan": None,
        "vat_schedules": []
    }
    result = {
        "status": status_str,
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, separators=(",", ":")))