import json
from ortools.linear_solver import pywraplp

inp = {
    "profits_per_kg": {
        "product_A": 30,
        "product_B": 10
    },
    "production_hours_per_kg": {
        "product_A": 6,
        "product_B": 3
    },
    "max_production_hours_per_week": 40,
    "min_output_ratio_B_to_A": 3,
    "storage_space_per_kg_relative_to_B": {
        "product_A": 4,
        "product_B": 1
    },
    "max_storage_capacity_equivalent_to_product_A_kg": 4
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver.")

x_a = solver.NumVar(0.0, solver.infinity(), "product_A_kg")
x_b = solver.NumVar(0.0, solver.infinity(), "product_B_kg")

profits = inp["profits_per_kg"]
hours = inp["production_hours_per_kg"]
storage = inp["storage_space_per_kg_relative_to_B"]

max_storage_units = (
    inp["max_storage_capacity_equivalent_to_product_A_kg"] * storage["product_A"]
)

solver.Add(
    hours["product_A"] * x_a + hours["product_B"] * x_b
    <= inp["max_production_hours_per_week"]
)
solver.Add(x_b >= inp["min_output_ratio_B_to_A"] * x_a)
solver.Add(storage["product_A"] * x_a + storage["product_B"] * x_b <= max_storage_units)

solver.Maximize(profits["product_A"] * x_a + profits["product_B"] * x_b)

status = solver.Solve()

def clean_number(value):
    value = float(value)
    if abs(value) < 1e-9:
        value = 0.0
    value = round(value, 6)
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "product_A_kg": clean_number(x_a.solution_value()),
        "product_B_kg": clean_number(x_b.solution_value()),
        "max_profit": clean_number(solver.Objective().Value())
    }
    objective_value = output["max_profit"]
else:
    output = {
        "product_A_kg": None,
        "product_B_kg": None,
        "max_profit": None
    }
    objective_value = None

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))