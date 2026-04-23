import json
from ortools.linear_solver import pywraplp

inp = {
    "products": [
        {
            "name": "high_end",
            "labor_hours_per_unit": 17,
            "inspection_hours_per_unit": 8,
            "profit_per_unit": 300,
            "max_demand": 50
        },
        {
            "name": "mid_range",
            "labor_hours_per_unit": 10,
            "inspection_hours_per_unit": 4,
            "profit_per_unit": 200,
            "max_demand": 80
        },
        {
            "name": "low_end",
            "labor_hours_per_unit": 2,
            "inspection_hours_per_unit": 2,
            "profit_per_unit": 100,
            "max_demand": 150
        }
    ],
    "resource_limits": {
        "labor_hours": 1000,
        "inspection_hours": 500
    }
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

products = {p["name"]: p for p in inp["products"]}

x_high = solver.IntVar(0, products["high_end"]["max_demand"], "high_end")
x_mid = solver.IntVar(0, products["mid_range"]["max_demand"], "mid_range")
x_low = solver.IntVar(0, products["low_end"]["max_demand"], "low_end")

solver.Add(
    products["high_end"]["labor_hours_per_unit"] * x_high
    + products["mid_range"]["labor_hours_per_unit"] * x_mid
    + products["low_end"]["labor_hours_per_unit"] * x_low
    <= inp["resource_limits"]["labor_hours"]
)

solver.Add(
    products["high_end"]["inspection_hours_per_unit"] * x_high
    + products["mid_range"]["inspection_hours_per_unit"] * x_mid
    + products["low_end"]["inspection_hours_per_unit"] * x_low
    <= inp["resource_limits"]["inspection_hours"]
)

profit_expr = (
    products["high_end"]["profit_per_unit"] * x_high
    + products["mid_range"]["profit_per_unit"] * x_mid
    + products["low_end"]["profit_per_unit"] * x_low
)

solver.Maximize(profit_expr)
status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

objective_value = None
output = {
    "production_plan": {
        "high_end": 0,
        "mid_range": 0,
        "low_end": 0
    }
}

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    objective_value = int(round(solver.Objective().Value()))

    # Tie-break among multiple optimal solutions: maximize low-end production.
    solver.Add(profit_expr == objective_value)
    obj = solver.Objective()
    obj.Clear()
    obj.SetCoefficient(x_low, 1)
    obj.SetMaximization()
    status = solver.Solve()

    output = {
        "production_plan": {
            "high_end": int(round(x_high.solution_value())),
            "mid_range": int(round(x_mid.solution_value())),
            "low_end": int(round(x_low.solution_value()))
        }
    }

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))