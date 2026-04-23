import json
from ortools.linear_solver import pywraplp

inp = {
    "restaurants": [
        {
            "name": "A",
            "annual_revenue": 15000,
            "cost_million": 1.6
        },
        {
            "name": "B",
            "annual_revenue": 40000,
            "cost_million": 2.5
        },
        {
            "name": "C",
            "annual_revenue": 30000,
            "cost_million": 1.8
        },
        {
            "name": "D",
            "annual_revenue": 50000,
            "cost_million": 3.0
        }
    ],
    "budget_million": 6.0,
    "mutual_exclusion_rules": [
        {
            "if_purchased": "D",
            "cannot_purchase": "A"
        }
    ]
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

restaurants = inp["restaurants"]
budget = inp["budget_million"]

x = {}
for r in restaurants:
    x[r["name"]] = solver.BoolVar(f"x_{r['name']}")

solver.Add(
    solver.Sum(r["cost_million"] * x[r["name"]] for r in restaurants) <= budget
)

for rule in inp["mutual_exclusion_rules"]:
    a = rule["if_purchased"]
    b = rule["cannot_purchase"]
    solver.Add(x[a] + x[b] <= 1)

objective = solver.Objective()
for r in restaurants:
    objective.SetCoefficient(x[r["name"]], r["annual_revenue"])
objective.SetMaximization()

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
    purchase_decisions = {r["name"]: int(round(x[r["name"]].solution_value())) for r in restaurants}
    selected_restaurants = [r["name"] for r in restaurants if purchase_decisions[r["name"]] == 1]
    total_cost_million = sum(r["cost_million"] * purchase_decisions[r["name"]] for r in restaurants)
    total_annual_revenue = sum(r["annual_revenue"] * purchase_decisions[r["name"]] for r in restaurants)

    output = {
        "selected_restaurants": selected_restaurants,
        "purchase_decisions": purchase_decisions,
        "total_cost_million": total_cost_million,
        "total_annual_revenue": total_annual_revenue
    }

    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": total_annual_revenue,
        "example_output": output
    }
else:
    output = {
        "selected_restaurants": [],
        "purchase_decisions": {r["name"]: 0 for r in restaurants},
        "total_cost_million": 0,
        "total_annual_revenue": 0
    }
    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))