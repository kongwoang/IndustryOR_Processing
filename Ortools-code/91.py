import json
from ortools.linear_solver import pywraplp

inp = {
    "products": [
        {
            "name": "A",
            "profit_per_unit": 3,
            "assembly_time_minutes": 12
        },
        {
            "name": "B",
            "profit_per_unit": 5,
            "assembly_time_minutes": 25
        }
    ],
    "weekly_machine_time_hours": 30,
    "min_B_per_A_ratio": {
        "B_units": 2,
        "A_units": 5
    }
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver.")

products = {p["name"]: p for p in inp["products"]}
A = solver.NumVar(0.0, solver.infinity(), "A")
B = solver.NumVar(0.0, solver.infinity(), "B")

available_minutes = inp["weekly_machine_time_hours"] * 60

solver.Add(
    products["A"]["assembly_time_minutes"] * A +
    products["B"]["assembly_time_minutes"] * B
    <= available_minutes
)

solver.Add(
    inp["min_B_per_A_ratio"]["A_units"] * B >=
    inp["min_B_per_A_ratio"]["B_units"] * A
)

solver.Maximize(
    products["A"]["profit_per_unit"] * A +
    products["B"]["profit_per_unit"] * B
)

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

def clean_number(x):
    return round(float(x), 6)

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "production": {
            "A": clean_number(A.solution_value()),
            "B": clean_number(B.solution_value())
        },
        "total_profit": clean_number(solver.Objective().Value())
    }
    objective_value = clean_number(solver.Objective().Value())
else:
    output = {
        "production": {
            "A": None,
            "B": None
        },
        "total_profit": None
    }
    objective_value = None

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))