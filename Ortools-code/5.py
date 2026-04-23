import json
from ortools.linear_solver import pywraplp

inp = {
    "proteins": {
        "salmon": {
            "price_per_100g": 4.0
        },
        "beef": {
            "price_per_100g": 3.6
        },
        "pork": {
            "price_per_100g": 1.8
        }
    },
    "vegetables": {
        "okra": {
            "price_per_100g": 2.6,
            "fiber_per_100g": 3.2
        },
        "carrots": {
            "price_per_100g": 1.2,
            "fiber_per_100g": 2.7
        },
        "celery": {
            "price_per_100g": 1.6,
            "fiber_per_100g": 1.6
        },
        "cabbage": {
            "price_per_100g": 2.3,
            "fiber_per_100g": 2.0
        }
    },
    "budget": 15.0,
    "total_food_grams": 600.0,
    "choose_exactly_one_protein": True,
    "min_selected_vegetables": 2,
    "min_grams_if_selected": 10.0
}

proteins = list(inp["proteins"].keys())
vegetables = list(inp["vegetables"].keys())
foods = proteins + vegetables

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

total_food_grams = inp["total_food_grams"]
min_grams_if_selected = inp["min_grams_if_selected"]

x = {f: solver.NumVar(0.0, total_food_grams, f"x_{f}") for f in foods}
y = {p: solver.IntVar(0, 1, f"y_{p}") for p in proteins}
z = {v: solver.IntVar(0, 1, f"z_{v}") for v in vegetables}

if inp["choose_exactly_one_protein"]:
    solver.Add(sum(y[p] for p in proteins) == 1)

for p in proteins:
    solver.Add(x[p] <= total_food_grams * y[p])
    solver.Add(x[p] >= min_grams_if_selected * y[p])

for v in vegetables:
    solver.Add(x[v] <= total_food_grams * z[v])
    solver.Add(x[v] >= min_grams_if_selected * z[v])

solver.Add(sum(z[v] for v in vegetables) >= inp["min_selected_vegetables"])
solver.Add(sum(x[f] for f in foods) == total_food_grams)

solver.Add(
    sum(inp["proteins"][p]["price_per_100g"] * x[p] / 100.0 for p in proteins) +
    sum(inp["vegetables"][v]["price_per_100g"] * x[v] / 100.0 for v in vegetables)
    <= inp["budget"]
)

objective = solver.Objective()
for v in vegetables:
    objective.SetCoefficient(x[v], inp["vegetables"][v]["fiber_per_100g"] / 100.0)
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

if result_status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    selected_protein = next(p for p in proteins if y[p].solution_value() > 0.5)

    food_quantities_grams = {
        f: round(x[f].solution_value(), 4) for f in foods
    }

    total_cost = (
        sum(inp["proteins"][p]["price_per_100g"] * x[p].solution_value() / 100.0 for p in proteins) +
        sum(inp["vegetables"][v]["price_per_100g"] * x[v].solution_value() / 100.0 for v in vegetables)
    )

    total_fiber_grams = sum(
        inp["vegetables"][v]["fiber_per_100g"] * x[v].solution_value() / 100.0 for v in vegetables
    )

    output = {
        "selected_protein": selected_protein,
        "food_quantities_grams": food_quantities_grams,
        "total_cost": round(total_cost, 4),
        "total_fiber_grams": round(total_fiber_grams, 4)
    }

    objective_value = round(solver.Objective().Value(), 4)
else:
    output = {
        "selected_protein": "",
        "food_quantities_grams": {f: 0.0 for f in foods},
        "total_cost": 0.0,
        "total_fiber_grams": 0.0
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(result_status, str(result_status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))