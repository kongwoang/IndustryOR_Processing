import json
from ortools.linear_solver import pywraplp

inp = {
    "products": [
        {
            "name": "A",
            "steel_kg_per_unit": 6,
            "aluminum_kg_per_unit": 8,
            "labor_hours_per_unit": 11,
            "profit_yuan_per_unit": 5000
        },
        {
            "name": "B",
            "steel_kg_per_unit": 12,
            "aluminum_kg_per_unit": 20,
            "labor_hours_per_unit": 24,
            "profit_yuan_per_unit": 11000
        }
    ],
    "resources": {
        "steel_kg": 200,
        "aluminum_kg": 300,
        "regular_labor_hours": 300
    },
    "overtime_cost_yuan_per_hour": 100
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("GLOP solver is not available.")

products = {p["name"]: p for p in inp["products"]}

A = solver.NumVar(0.0, solver.infinity(), "A")
B = solver.NumVar(0.0, solver.infinity(), "B")
overtime = solver.NumVar(0.0, solver.infinity(), "overtime")

steel_used = products["A"]["steel_kg_per_unit"] * A + products["B"]["steel_kg_per_unit"] * B
aluminum_used = products["A"]["aluminum_kg_per_unit"] * A + products["B"]["aluminum_kg_per_unit"] * B
labor_used = products["A"]["labor_hours_per_unit"] * A + products["B"]["labor_hours_per_unit"] * B
gross_profit = products["A"]["profit_yuan_per_unit"] * A + products["B"]["profit_yuan_per_unit"] * B

solver.Add(steel_used <= inp["resources"]["steel_kg"])
solver.Add(aluminum_used <= inp["resources"]["aluminum_kg"])
solver.Add(labor_used <= inp["resources"]["regular_labor_hours"] + overtime)

# Phase 1: minimize overtime
solver.Minimize(overtime)
status1 = solver.Solve()

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
        "A": None,
        "B": None
    },
    "resource_usage": {
        "steel_kg": None,
        "aluminum_kg": None,
        "labor_hours": None,
        "overtime_hours": None
    },
    "profit_breakdown": {
        "gross_profit_yuan": None,
        "overtime_cost_yuan": None,
        "net_profit_yuan": None
    }
}

if status1 in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    min_overtime = overtime.solution_value()

    # Phase 2: maximize profit subject to minimal overtime
    solver.Add(overtime <= min_overtime + 1e-9)
    solver.Maximize(gross_profit)

    status2 = solver.Solve()

    if status2 in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        a_val = A.solution_value()
        b_val = B.solution_value()
        overtime_val = overtime.solution_value()

        steel_val = products["A"]["steel_kg_per_unit"] * a_val + products["B"]["steel_kg_per_unit"] * b_val
        aluminum_val = products["A"]["aluminum_kg_per_unit"] * a_val + products["B"]["aluminum_kg_per_unit"] * b_val
        labor_val = products["A"]["labor_hours_per_unit"] * a_val + products["B"]["labor_hours_per_unit"] * b_val
        gross_val = products["A"]["profit_yuan_per_unit"] * a_val + products["B"]["profit_yuan_per_unit"] * b_val
        overtime_cost_val = inp["overtime_cost_yuan_per_hour"] * overtime_val
        net_val = gross_val - overtime_cost_val

        objective_value = net_val
        output = {
            "production_plan": {
                "A": a_val,
                "B": b_val
            },
            "resource_usage": {
                "steel_kg": steel_val,
                "aluminum_kg": aluminum_val,
                "labor_hours": labor_val,
                "overtime_hours": overtime_val
            },
            "profit_breakdown": {
                "gross_profit_yuan": gross_val,
                "overtime_cost_yuan": overtime_cost_val,
                "net_profit_yuan": net_val
            }
        }
        final_status = status2
    else:
        final_status = status2
else:
    final_status = status1

result = {
    "status": status_map.get(final_status, str(final_status)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))