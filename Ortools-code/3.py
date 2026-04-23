import json
from ortools.linear_solver import pywraplp

inp = {
    "months": ["January", "February", "March", "April", "May", "June"],
    "demand": {
        "January": 20000,
        "February": 40000,
        "March": 42000,
        "April": 35000,
        "May": 19000,
        "June": 18500
    },
    "initial_workforce": 1000,
    "initial_inventory": 15000,
    "initial_backorder": 0,
    "ending_inventory_min": 10000,
    "ending_backorder_required": 0,
    "sales_price_per_unit": 300,
    "raw_material_cost_per_inhouse_unit": 90,
    "outsourcing_cost_per_unit": 200,
    "inventory_holding_cost_per_unit_month": 15,
    "backorder_cost_per_unit_month": 35,
    "labor_hours_per_unit": 5,
    "regular_hours_per_worker_month": 160,
    "regular_wage_per_hour": 30,
    "max_overtime_hours_per_worker_month": 20,
    "overtime_wage_per_hour": 40,
    "hire_cost_per_worker": 5000,
    "fire_cost_per_worker": 8000
}

months = inp["months"]
demand = inp["demand"]

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

# Decision variables
workforce = {}
hires = {}
fires = {}
inhouse = {}
outsourcing = {}
overtime = {}
inventory = {}
backorder = {}

for m in months:
    workforce[m] = solver.IntVar(0, solver.infinity(), f"workforce_{m}")
    hires[m] = solver.IntVar(0, solver.infinity(), f"hires_{m}")
    fires[m] = solver.IntVar(0, solver.infinity(), f"fires_{m}")
    inhouse[m] = solver.IntVar(0, solver.infinity(), f"inhouse_{m}")
    outsourcing[m] = solver.IntVar(0, solver.infinity(), f"outsourcing_{m}")
    overtime[m] = solver.NumVar(0, solver.infinity(), f"overtime_{m}")
    inventory[m] = solver.IntVar(0, solver.infinity(), f"inventory_{m}")
    backorder[m] = solver.IntVar(0, solver.infinity(), f"backorder_{m}")

# Workforce balance
for i, m in enumerate(months):
    if i == 0:
        prev_workforce = inp["initial_workforce"]
    else:
        prev_workforce = workforce[months[i - 1]]
    solver.Add(workforce[m] == prev_workforce + hires[m] - fires[m])

# Production capacity and overtime limits
for m in months:
    solver.Add(inp["labor_hours_per_unit"] * inhouse[m] <= inp["regular_hours_per_worker_month"] * workforce[m] + overtime[m])
    solver.Add(overtime[m] <= inp["max_overtime_hours_per_worker_month"] * workforce[m])

# Inventory / backorder flow:
# inventory_t - backorder_t = inventory_{t-1} - backorder_{t-1} + inhouse_t + outsourcing_t - demand_t
for i, m in enumerate(months):
    if i == 0:
        prev_net = inp["initial_inventory"] - inp["initial_backorder"]
    else:
        prev_m = months[i - 1]
        prev_net = inventory[prev_m] - backorder[prev_m]
    solver.Add(inventory[m] - backorder[m] == prev_net + inhouse[m] + outsourcing[m] - demand[m])

# Terminal conditions
last_month = months[-1]
solver.Add(inventory[last_month] >= inp["ending_inventory_min"])
solver.Add(backorder[last_month] == inp["ending_backorder_required"])

# Objective: maximize total net profit
total_revenue = solver.Sum(
    inp["sales_price_per_unit"] * (demand[m] + (backorder[months[i - 1]] if i > 0 else inp["initial_backorder"]) - backorder[m])
    for i, m in enumerate(months)
)

total_cost = solver.Sum(
    inp["raw_material_cost_per_inhouse_unit"] * inhouse[m]
    + inp["outsourcing_cost_per_unit"] * outsourcing[m]
    + inp["inventory_holding_cost_per_unit_month"] * inventory[m]
    + inp["backorder_cost_per_unit_month"] * backorder[m]
    + inp["regular_wage_per_hour"] * inp["regular_hours_per_worker_month"] * workforce[m]
    + inp["overtime_wage_per_hour"] * overtime[m]
    + inp["hire_cost_per_worker"] * hires[m]
    + inp["fire_cost_per_worker"] * fires[m]
    for m in months
)

solver.Maximize(total_revenue - total_cost)

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

output = {
    "monthly_plan": [],
    "totals": {
        "total_revenue": None,
        "total_cost": None,
        "net_profit": None
    }
}

objective_value = None

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    total_revenue_value = 0.0
    total_cost_value = 0.0

    for i, m in enumerate(months):
        prev_backorder = inp["initial_backorder"] if i == 0 else backorder[months[i - 1]].solution_value()
        units_sold = demand[m] + prev_backorder - backorder[m].solution_value()

        monthly_revenue = inp["sales_price_per_unit"] * units_sold
        monthly_cost = (
            inp["raw_material_cost_per_inhouse_unit"] * inhouse[m].solution_value()
            + inp["outsourcing_cost_per_unit"] * outsourcing[m].solution_value()
            + inp["inventory_holding_cost_per_unit_month"] * inventory[m].solution_value()
            + inp["backorder_cost_per_unit_month"] * backorder[m].solution_value()
            + inp["regular_wage_per_hour"] * inp["regular_hours_per_worker_month"] * workforce[m].solution_value()
            + inp["overtime_wage_per_hour"] * overtime[m].solution_value()
            + inp["hire_cost_per_worker"] * hires[m].solution_value()
            + inp["fire_cost_per_worker"] * fires[m].solution_value()
        )
        monthly_profit = monthly_revenue - monthly_cost

        total_revenue_value += monthly_revenue
        total_cost_value += monthly_cost

        output["monthly_plan"].append({
            "month": m,
            "workforce": int(round(workforce[m].solution_value())),
            "hires": int(round(hires[m].solution_value())),
            "fires": int(round(fires[m].solution_value())),
            "inhouse_production": int(round(inhouse[m].solution_value())),
            "outsourcing": int(round(outsourcing[m].solution_value())),
            "overtime_hours": round(overtime[m].solution_value(), 2),
            "ending_inventory": int(round(inventory[m].solution_value())),
            "ending_backorder": int(round(backorder[m].solution_value())),
            "units_sold": int(round(units_sold)),
            "monthly_revenue": round(monthly_revenue, 2),
            "monthly_cost": round(monthly_cost, 2),
            "monthly_profit": round(monthly_profit, 2)
        })

    net_profit_value = total_revenue_value - total_cost_value
    output["totals"] = {
        "total_revenue": round(total_revenue_value, 2),
        "total_cost": round(total_cost_value, 2),
        "net_profit": round(net_profit_value, 2)
    }
    objective_value = round(solver.Objective().Value(), 2)

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))