import json
from ortools.linear_solver import pywraplp

inp = {
    "months": [1, 2, 3],
    "initial_inventory": 200,
    "warehouse_capacity": 500,
    "purchase_price": [8, 6, 9],
    "selling_price": [9, 8, 10]
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("OR-Tools GLOP solver is not available.")

T = len(inp["months"])
capacity = inp["warehouse_capacity"]
initial_inventory = inp["initial_inventory"]
purchase_price = inp["purchase_price"]
selling_price = inp["selling_price"]

purchase = [solver.NumVar(0.0, solver.infinity(), f"purchase_{t+1}") for t in range(T)]
sales = [solver.NumVar(0.0, solver.infinity(), f"sales_{t+1}") for t in range(T)]
inventory = [solver.NumVar(0.0, solver.infinity(), f"inventory_{t+1}") for t in range(T)]

for t in range(T):
    prev_inventory = initial_inventory if t == 0 else inventory[t - 1]
    solver.Add(prev_inventory + purchase[t] <= capacity)
    solver.Add(inventory[t] == prev_inventory + purchase[t] - sales[t])
    solver.Add(inventory[t] <= capacity)

objective = solver.Objective()
for t in range(T):
    objective.SetCoefficient(sales[t], selling_price[t])
    objective.SetCoefficient(purchase[t], -purchase_price[t])
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
    if x is None:
        return None
    if abs(x - round(x)) <= 1e-9:
        return int(round(x))
    return float(x)

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    objective_value = clean(objective.Value())
    output = {
        "purchase_plan": [clean(v.solution_value()) for v in purchase],
        "sales_plan": [clean(v.solution_value()) for v in sales],
        "ending_inventory": [clean(v.solution_value()) for v in inventory],
        "total_profit": objective_value
    }
else:
    objective_value = None
    output = {
        "purchase_plan": [None] * T,
        "sales_plan": [None] * T,
        "ending_inventory": [None] * T,
        "total_profit": None
    }

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))