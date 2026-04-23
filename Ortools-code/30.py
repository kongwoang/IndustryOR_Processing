import json
from ortools.linear_solver import pywraplp

inp = {
    "requirements": {
        "protein_g_min": 700,
        "minerals_g_min": 30,
        "vitamins_mg_min": 100
    },
    "feeds": [
        {
            "feed_id": 1,
            "protein_g_per_kg": 3,
            "minerals_g_per_kg": 1,
            "vitamins_mg_per_kg": 0.5,
            "price_yen_per_kg": 0.2
        },
        {
            "feed_id": 2,
            "protein_g_per_kg": 2,
            "minerals_g_per_kg": 0.5,
            "vitamins_mg_per_kg": 1,
            "price_yen_per_kg": 0.7
        },
        {
            "feed_id": 3,
            "protein_g_per_kg": 1,
            "minerals_g_per_kg": 0.2,
            "vitamins_mg_per_kg": 0.2,
            "price_yen_per_kg": 0.4
        },
        {
            "feed_id": 4,
            "protein_g_per_kg": 6,
            "minerals_g_per_kg": 2,
            "vitamins_mg_per_kg": 2,
            "price_yen_per_kg": 0.3
        },
        {
            "feed_id": 5,
            "protein_g_per_kg": 18,
            "minerals_g_per_kg": 0.5,
            "vitamins_mg_per_kg": 0.8,
            "price_yen_per_kg": 0.8
        }
    ]
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver.")

x = {}
for feed in inp["feeds"]:
    fid = feed["feed_id"]
    x[fid] = solver.NumVar(0.0, solver.infinity(), f"x_{fid}")

req = inp["requirements"]

solver.Add(
    sum(feed["protein_g_per_kg"] * x[feed["feed_id"]] for feed in inp["feeds"])
    >= req["protein_g_min"]
)
solver.Add(
    sum(feed["minerals_g_per_kg"] * x[feed["feed_id"]] for feed in inp["feeds"])
    >= req["minerals_g_min"]
)
solver.Add(
    sum(feed["vitamins_mg_per_kg"] * x[feed["feed_id"]] for feed in inp["feeds"])
    >= req["vitamins_mg_min"]
)

objective = solver.Objective()
for feed in inp["feeds"]:
    objective.SetCoefficient(x[feed["feed_id"]], feed["price_yen_per_kg"])
objective.SetMinimization()

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
    total_cost = round(objective.Value(), 6)
    output = {
        "feed_amounts_kg": [
            {
                "feed_id": feed["feed_id"],
                "amount_kg": round(x[feed["feed_id"]].solution_value(), 6)
            }
            for feed in inp["feeds"]
        ],
        "total_cost_yen": total_cost
    }
    objective_value = total_cost
else:
    output = {
        "feed_amounts_kg": [
            {
                "feed_id": feed["feed_id"],
                "amount_kg": 0.0
            }
            for feed in inp["feeds"]
        ],
        "total_cost_yen": None
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))