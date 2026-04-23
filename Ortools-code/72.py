import json
from ortools.linear_solver import pywraplp

inp = {
    "properties": [
        {
            "id": 1,
            "annual_income": 12500,
            "cost_million": 1.5
        },
        {
            "id": 2,
            "annual_income": 35000,
            "cost_million": 2.1
        },
        {
            "id": 3,
            "annual_income": 23000,
            "cost_million": 2.3
        },
        {
            "id": 4,
            "annual_income": 100000,
            "cost_million": 4.2
        }
    ],
    "budget_million": 7.0,
    "mutually_exclusive_pairs": [
        [4, 3]
    ]
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

properties = inp["properties"]
budget = inp["budget_million"]

x = {}
for p in properties:
    pid = p["id"]
    x[pid] = solver.BoolVar(f"x_{pid}")

solver.Add(
    solver.Sum(p["cost_million"] * x[p["id"]] for p in properties) <= budget
)

for a, b in inp["mutually_exclusive_pairs"]:
    solver.Add(x[a] + x[b] <= 1)

solver.Maximize(
    solver.Sum(p["annual_income"] * x[p["id"]] for p in properties)
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

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    selected_properties = [p["id"] for p in properties if x[p["id"]].solution_value() > 0.5]
    buy = {f"property_{p['id']}": int(x[p["id"]].solution_value() > 0.5) for p in properties}
    total_cost_million = sum(p["cost_million"] * buy[f"property_{p['id']}"] for p in properties)
    total_annual_income = sum(p["annual_income"] * buy[f"property_{p['id']}"] for p in properties)

    output = {
        "selected_properties": selected_properties,
        "buy": buy,
        "total_cost_million": total_cost_million,
        "total_annual_income": total_annual_income
    }
    objective_value = solver.Objective().Value()
else:
    output = {
        "selected_properties": [],
        "buy": {f"property_{p['id']}": 0 for p in properties},
        "total_cost_million": 0,
        "total_annual_income": 0
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))