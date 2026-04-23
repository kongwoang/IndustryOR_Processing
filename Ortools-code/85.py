import json
from ortools.linear_solver import pywraplp

inp = {
    "products": ["X", "Y"],
    "machine_time_minutes_per_batch": {
        "X": 13,
        "Y": 19
    },
    "craftsman_time_minutes_per_batch": {
        "X": 20,
        "Y": 29
    },
    "available_machine_hours_per_week": 40,
    "available_craftsman_hours_per_week": 35,
    "machine_cost_per_hour": 10,
    "craftsman_cost_per_hour": 2,
    "revenue_per_batch": {
        "X": 20,
        "Y": 30
    },
    "min_batches": {
        "X": 10,
        "Y": 0
    },
    "allow_fractional_batches": True
}

solver = pywraplp.Solver.CreateSolver("GLOP")

products = inp["products"]
machine_capacity = inp["available_machine_hours_per_week"] * 60.0
craft_capacity = inp["available_craftsman_hours_per_week"] * 60.0

x = {}
for p in products:
    lb = float(inp["min_batches"].get(p, 0))
    x[p] = solver.NumVar(lb, solver.infinity(), f"x_{p}")

solver.Add(
    sum(inp["machine_time_minutes_per_batch"][p] * x[p] for p in products) <= machine_capacity
)
solver.Add(
    sum(inp["craftsman_time_minutes_per_batch"][p] * x[p] for p in products) <= craft_capacity
)

objective = solver.Objective()
for p in products:
    machine_cost_per_batch = inp["machine_time_minutes_per_batch"][p] * inp["machine_cost_per_hour"] / 60.0
    craftsman_cost_per_batch = inp["craftsman_time_minutes_per_batch"][p] * inp["craftsman_cost_per_hour"] / 60.0
    profit_per_batch = inp["revenue_per_batch"][p] - machine_cost_per_batch - craftsman_cost_per_batch
    objective.SetCoefficient(x[p], profit_per_batch)
objective.SetMaximization()

result_status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

production = {p: x[p].solution_value() for p in products}
machine_used = sum(inp["machine_time_minutes_per_batch"][p] * production[p] for p in products)
craft_used = sum(inp["craftsman_time_minutes_per_batch"][p] * production[p] for p in products)

output = {
    "production": {
        "X": production["X"],
        "Y": production["Y"]
    },
    "resource_usage": {
        "machine_minutes_used": machine_used,
        "craftsman_minutes_used": craft_used,
        "machine_minutes_idle": machine_capacity - machine_used,
        "craftsman_minutes_idle": craft_capacity - craft_used
    }
}

result = {
    "status": status_map.get(result_status, str(result_status)),
    "objective_value": round(solver.Objective().Value(), 2) if result_status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else None,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))