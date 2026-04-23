import json
from ortools.linear_solver import pywraplp

inp = {
    "warehouse_capacity": 5000,
    "initial_inventory": 1000,
    "initial_funds": 20000,
    "target_ending_inventory": 2000,
    "months": [1, 2, 3],
    "purchase_prices": {
        "1": 2.85,
        "2": 3.05,
        "3": 2.9
    },
    "selling_prices": {
        "1": 3.1,
        "2": 3.25,
        "3": 2.95
    }
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver.")

months = [str(m) for m in inp["months"]]
capacity = inp["warehouse_capacity"]
initial_inventory = inp["initial_inventory"]
initial_funds = inp["initial_funds"]
target_ending_inventory = inp["target_ending_inventory"]
purchase_prices = inp["purchase_prices"]
selling_prices = inp["selling_prices"]

buy = {m: solver.NumVar(0.0, solver.infinity(), f"buy_{m}") for m in months}
sell = {m: solver.NumVar(0.0, solver.infinity(), f"sell_{m}") for m in months}
inventory = {m: solver.NumVar(0.0, capacity, f"inventory_{m}") for m in months}
cash = {m: solver.NumVar(0.0, solver.infinity(), f"cash_{m}") for m in months}

# Inventory balance
solver.Add(inventory["1"] == initial_inventory - sell["1"] + buy["1"])
solver.Add(inventory["2"] == inventory["1"] - sell["2"] + buy["2"])
solver.Add(inventory["3"] == inventory["2"] - sell["3"] + buy["3"])

# Can only sell what is available at the beginning of each month
solver.Add(sell["1"] <= initial_inventory)
solver.Add(sell["2"] <= inventory["1"])
solver.Add(sell["3"] <= inventory["2"])

# Cash balance
solver.Add(cash["1"] == initial_funds + selling_prices["1"] * sell["1"] - purchase_prices["1"] * buy["1"])
solver.Add(cash["2"] == cash["1"] + selling_prices["2"] * sell["2"] - purchase_prices["2"] * buy["2"])
solver.Add(cash["3"] == cash["2"] + selling_prices["3"] * sell["3"] - purchase_prices["3"] * buy["3"])

# No borrowing
solver.Add(cash["1"] >= 0)
solver.Add(cash["2"] >= 0)
solver.Add(cash["3"] >= 0)

# Ending inventory requirement
solver.Add(inventory["3"] == target_ending_inventory)

# Profit = final cash - initial funds
profit = cash["3"] - initial_funds
solver.Maximize(profit)

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

def clean(x):
    if abs(x - round(x)) < 1e-9:
        return float(round(x))
    return round(x, 6)

output = {
    "purchase_plan": {
        "month_1": clean(buy["1"].solution_value()),
        "month_2": clean(buy["2"].solution_value()),
        "month_3": clean(buy["3"].solution_value())
    },
    "sales_plan": {
        "month_1": clean(sell["1"].solution_value()),
        "month_2": clean(sell["2"].solution_value()),
        "month_3": clean(sell["3"].solution_value())
    },
    "ending_inventory_by_month": {
        "month_1": clean(inventory["1"].solution_value()),
        "month_2": clean(inventory["2"].solution_value()),
        "month_3": clean(inventory["3"].solution_value())
    },
    "ending_cash_by_month": {
        "month_1": clean(cash["1"].solution_value()),
        "month_2": clean(cash["2"].solution_value()),
        "month_3": clean(cash["3"].solution_value())
    },
    "total_sales_revenue": clean(
        sum(selling_prices[m] * sell[m].solution_value() for m in months)
    ),
    "total_purchase_cost": clean(
        sum(purchase_prices[m] * buy[m].solution_value() for m in months)
    ),
    "final_cash": clean(cash["3"].solution_value())
}

print(json.dumps({
    "status": status,
    "objective_value": clean(solver.Objective().Value()),  # profit
    "example_output": output
}, ensure_ascii=False))