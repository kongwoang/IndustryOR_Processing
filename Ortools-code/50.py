import json
from ortools.linear_solver import pywraplp

inp = {
    "months": ["7", "8", "9", "10", "11", "12"],
    "buy_prices": [28, 24, 25, 27, 23, 23],
    "sell_prices": [29, 24, 26, 28, 22, 25],
    "initial_inventory": 200,
    "warehouse_capacity": 500,
    "purchases_at_beginning_of_month": True
}

months = inp["months"]
buy_prices = inp["buy_prices"]
sell_prices = inp["sell_prices"]
initial_inventory = inp["initial_inventory"]
capacity = inp["warehouse_capacity"]
n = len(months)

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver.")

buy = [solver.NumVar(0.0, solver.infinity(), f"buy_{t}") for t in range(n)]
sell = [solver.NumVar(0.0, solver.infinity(), f"sell_{t}") for t in range(n)]
inv = [solver.NumVar(0.0, capacity, f"inv_{t}") for t in range(n)]

for t in range(n):
    if t == 0:
        solver.Add(initial_inventory + buy[t] <= capacity)
        solver.Add(inv[t] == initial_inventory + buy[t] - sell[t])
    else:
        solver.Add(inv[t - 1] + buy[t] <= capacity)
        solver.Add(inv[t] == inv[t - 1] + buy[t] - sell[t])

objective = solver.Objective()
for t in range(n):
    objective.SetCoefficient(sell[t], sell_prices[t])
    objective.SetCoefficient(buy[t], -buy_prices[t])
objective.SetMaximization()

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

def clean(x):
    if abs(x - round(x)) <= 1e-9:
        return int(round(x))
    return round(x, 6)

output = {
    "months": months,
    "purchase": [clean(buy[t].solution_value()) for t in range(n)] if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else [],
    "sales": [clean(sell[t].solution_value()) for t in range(n)] if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else [],
    "ending_inventory": [clean(inv[t].solution_value()) for t in range(n)] if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else []
}

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": clean(objective.Value()) if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else None,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))