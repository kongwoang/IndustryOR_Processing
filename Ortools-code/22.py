import json
from ortools.linear_solver import pywraplp

inp = {
    "products": ["A1", "A2", "A3"],
    "days_available": 22,
    "max_demand": {
        "A1": 5300,
        "A2": 4500,
        "A3": 5400
    },
    "selling_price": {
        "A1": 124,
        "A2": 109,
        "A3": 115
    },
    "production_cost": {
        "A1": 73.3,
        "A2": 52.9,
        "A3": 65.4
    },
    "daily_quota": {
        "A1": 500,
        "A2": 450,
        "A3": 550
    },
    "activation_cost": {
        "A1": 170000,
        "A2": 150000,
        "A3": 100000
    },
    "minimum_batch": {
        "A1": 20,
        "A2": 20,
        "A3": 16
    },
    "all_products_must_be_produced": True,
    "unit": "100kg"
}

products = inp["products"]
solver = pywraplp.Solver.CreateSolver("GLOP")

x = {
    p: solver.NumVar(inp["minimum_batch"][p], inp["max_demand"][p], f"x_{p}")
    for p in products
}

solver.Add(
    solver.Sum(x[p] / inp["daily_quota"][p] for p in products) <= inp["days_available"]
)

objective = solver.Objective()
for p in products:
    unit_margin = inp["selling_price"][p] - inp["production_cost"][p]
    objective.SetCoefficient(x[p], unit_margin)
objective.SetMaximization()

status_code = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}
status = status_map.get(status_code, "UNKNOWN")

if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    production_units = {p: x[p].solution_value() for p in products}
    activate = {p: 1 for p in products}
    days_used = sum(production_units[p] / inp["daily_quota"][p] for p in products)
    total_sales_revenue = sum(
        production_units[p] * inp["selling_price"][p] for p in products
    )
    total_variable_cost = sum(
        production_units[p] * inp["production_cost"][p] for p in products
    )
    total_fixed_activation_cost = sum(inp["activation_cost"][p] for p in products)
    net_profit = total_sales_revenue - total_variable_cost - total_fixed_activation_cost

    output = {
        "production_units": {p: round(production_units[p], 10) for p in products},
        "activate": activate,
        "days_used": round(days_used, 10),
        "days_unused": round(inp["days_available"] - days_used, 10),
        "total_sales_revenue": round(total_sales_revenue, 10),
        "total_variable_cost": round(total_variable_cost, 10),
        "total_fixed_activation_cost": round(total_fixed_activation_cost, 10),
        "net_profit": round(net_profit, 10)
    }
    objective_value = round(net_profit, 10)
else:
    output = {
        "production_units": {p: 0.0 for p in products},
        "activate": {p: 0 for p in products},
        "days_used": 0.0,
        "days_unused": float(inp["days_available"]),
        "total_sales_revenue": 0.0,
        "total_variable_cost": 0.0,
        "total_fixed_activation_cost": 0.0,
        "net_profit": 0.0
    }
    objective_value = None

print(json.dumps({
    "status": status,
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))