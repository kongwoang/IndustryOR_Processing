import json
from ortools.linear_solver import pywraplp

inp = {
    "quarters": ["Winter", "Spring", "Summer", "Autumn"],
    "purchase_price": [410, 430, 460, 450],
    "sale_price": [425, 440, 465, 455],
    "max_sales": [100, 140, 200, 160],
    "warehouse_capacity": 20,
    "storage_cost": {
        "a": 70,
        "b": 100
    }
}

quarters = inp["quarters"]
purchase_price = inp["purchase_price"]
sale_price = inp["sale_price"]
max_sales = inp["max_sales"]
capacity = inp["warehouse_capacity"]
a = inp["storage_cost"]["a"]
b = inp["storage_cost"]["b"]
n = len(quarters)

solver = pywraplp.Solver.CreateSolver("GLOP")

# x[i,j]: quantity bought in quarter i and sold in quarter j
x = {}
for i in range(n):
    for j in range(i, n):
        x[i, j] = solver.NumVar(0, solver.infinity(), f"x_{i}_{j}")

# Sales limit in each quarter
for j in range(n):
    solver.Add(sum(x[i, j] for i in range(j + 1)) <= max_sales[j])

# Storage capacity: only carry-over inventory counts
for t in range(n):
    solver.Add(sum(x[i, j] for (i, j) in x if i <= t < j) <= capacity)

# Objective
objective = solver.Objective()
for i in range(n):
    for j in range(i, n):
        u = j - i
        storage = 0 if u == 0 else (a + b * u)
        coeff = sale_price[j] - purchase_price[i] - storage
        objective.SetCoefficient(x[i, j], coeff)
objective.SetMaximization()

status = solver.Solve()

buy_to_sell_flow = {
    quarters[i]: {
        quarters[j]: round(x[i, j].solution_value(), 6) if i <= j else 0.0
        for j in range(n)
    }
    for i in range(n)
}

purchase_by_quarter = {
    quarters[i]: round(sum(x[i, j].solution_value() for j in range(i, n)), 6)
    for i in range(n)
}

sales_by_quarter = {
    quarters[j]: round(sum(x[i, j].solution_value() for i in range(j + 1)), 6)
    for j in range(n)
}

ending_inventory = {
    quarters[t]: round(sum(x[i, j].solution_value() for (i, j) in x if i <= t < j), 6)
    for t in range(n)
}

output = {
    "buy_to_sell_flow": buy_to_sell_flow,
    "purchase_by_quarter": purchase_by_quarter,
    "sales_by_quarter": sales_by_quarter,
    "ending_inventory": ending_inventory
}

print(json.dumps({
    "status": "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else str(status),
    "objective_value": round(objective.Value(), 6),
    "example_output": output
}, ensure_ascii=False))