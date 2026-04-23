import json
from ortools.linear_solver import pywraplp

inp = {
    "n": 10,
    "requirements": [3, 5, 2, 4, 6, 5, 4, 3, 2, 1],
    "costs": {
        "buy_new": 10,
        "slow_repair": 1,
        "fast_repair": 3
    },
    "repair_times": {
        "slow": 3,
        "fast": 1
    }
}

n = inp["n"]
r = inp["requirements"]
a = inp["costs"]["buy_new"]
b = inp["costs"]["slow_repair"]
c = inp["costs"]["fast_repair"]
p = inp["repair_times"]["slow"]
q = inp["repair_times"]["fast"]

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    solver = pywraplp.Solver.CreateSolver("CBC")
if solver is None:
    raise RuntimeError("No suitable MIP solver (SCIP/CBC) is available in OR-Tools.")

inf = solver.infinity()

buy = [solver.IntVar(0, inf, f"buy_{t}") for t in range(n)]
slow = [solver.IntVar(0, inf, f"slow_{t}") for t in range(n)]
fast = [solver.IntVar(0, inf, f"fast_{t}") for t in range(n)]
inventory = [solver.IntVar(0, inf, f"inventory_{t}") for t in range(n + 1)]

solver.Add(inventory[0] == 0)

for t in range(n):
    solver.Add(inventory[t] + buy[t] >= r[t])
    solver.Add(slow[t] + fast[t] <= r[t])

    if t + p + 1 >= n:
        solver.Add(slow[t] == 0)
    if t + q + 1 >= n:
        solver.Add(fast[t] == 0)

    returned = 0
    if t - p >= 0:
        returned += slow[t - p]
    if t - q >= 0:
        returned += fast[t - q]

    solver.Add(inventory[t + 1] == inventory[t] + buy[t] - r[t] + returned)

objective = solver.Objective()
for t in range(n):
    objective.SetCoefficient(buy[t], a)
    objective.SetCoefficient(slow[t], b)
    objective.SetCoefficient(fast[t], c)
objective.SetMinimization()

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
}

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "buy": [int(round(v.solution_value())) for v in buy],
        "slow_repair": [int(round(v.solution_value())) for v in slow],
        "fast_repair": [int(round(v.solution_value())) for v in fast],
        "inventory_start": [int(round(inventory[t].solution_value())) for t in range(n)],
        "inventory_after_stage": [int(round(inventory[t + 1].solution_value())) for t in range(n)]
    }
    objective_value = int(round(objective.Value()))
else:
    output = {
        "buy": [0] * n,
        "slow_repair": [0] * n,
        "fast_repair": [0] * n,
        "inventory_start": [0] * n,
        "inventory_after_stage": [0] * n
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))