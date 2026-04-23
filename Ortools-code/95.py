import json
from ortools.linear_solver import pywraplp

inp = {
    "initial_inventory_1000_boxes": 0,
    "storage_cost_1000_yuan_per_1000_boxes_per_week": 0.2,
    "weeks": [
        {
            "week": 1,
            "demand_1000_boxes": 15,
            "production_capacity_1000_boxes": 30,
            "production_cost_1000_yuan_per_1000_boxes": 5.0
        },
        {
            "week": 2,
            "demand_1000_boxes": 25,
            "production_capacity_1000_boxes": 40,
            "production_cost_1000_yuan_per_1000_boxes": 5.1
        },
        {
            "week": 3,
            "demand_1000_boxes": 35,
            "production_capacity_1000_boxes": 45,
            "production_cost_1000_yuan_per_1000_boxes": 5.4
        },
        {
            "week": 4,
            "demand_1000_boxes": 25,
            "production_capacity_1000_boxes": 20,
            "production_cost_1000_yuan_per_1000_boxes": 5.5
        }
    ]
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver.")

n = len(inp["weeks"])
prod = []
inv = []

for i, w in enumerate(inp["weeks"]):
    prod.append(solver.NumVar(0.0, w["production_capacity_1000_boxes"], f"prod_{i+1}"))
    inv.append(solver.NumVar(0.0, solver.infinity(), f"inv_{i+1}"))

initial_inventory = inp["initial_inventory_1000_boxes"]

for i, w in enumerate(inp["weeks"]):
    demand = w["demand_1000_boxes"]
    if i == 0:
        solver.Add(initial_inventory + prod[i] - inv[i] == demand)
    else:
        solver.Add(inv[i - 1] + prod[i] - inv[i] == demand)

solver.Add(inv[-1] == 0)

objective = solver.Objective()
for i, w in enumerate(inp["weeks"]):
    objective.SetCoefficient(prod[i], w["production_cost_1000_yuan_per_1000_boxes"])
for i in range(n - 1):
    objective.SetCoefficient(inv[i], inp["storage_cost_1000_yuan_per_1000_boxes_per_week"])
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

def clean(v):
    if abs(v - round(v)) <= 1e-9:
        return int(round(v))
    return round(v, 6)

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "production_plan": [
            {
                "week": inp["weeks"][i]["week"],
                "production_1000_boxes": clean(prod[i].solution_value()),
                "ending_inventory_1000_boxes": clean(inv[i].solution_value())
            }
            for i in range(n)
        ],
        "total_cost_1000_yuan": clean(objective.Value())
    }
    objective_value = clean(objective.Value())
else:
    output = {
        "production_plan": [],
        "total_cost_1000_yuan": None
    }
    objective_value = None

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))