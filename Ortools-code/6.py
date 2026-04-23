# Source problem statement: :contentReference[oaicite:0]{index=0}
import json
from ortools.linear_solver import pywraplp

inp = {
    "products": ["I", "II", "III"],
    "quarters": [1, 2, 3, 4],
    "demand": {
        "I": [1500, 1000, 2000, 1200],
        "II": [1500, 1500, 1200, 1500],
        "III": [1000, 2000, 1500, 2500]
    },
    "initial_inventory": {
        "I": 0,
        "II": 0,
        "III": 0
    },
    "required_ending_inventory": {
        "I": 150,
        "II": 150,
        "III": 150
    },
    "production_hours_per_quarter": 15000,
    "hours_per_unit": {
        "I": 2,
        "II": 4,
        "III": 3
    },
    "production_forbidden": [
        {
            "product": "I",
            "quarter": 2
        }
    ],
    "delay_penalty_per_unit_per_quarter": {
        "I": 20,
        "II": 20,
        "III": 10
    },
    "inventory_cost_per_unit_per_quarter": 5
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    solver = pywraplp.Solver.CreateSolver("CBC")

if solver is None:
    print(json.dumps({
        "status": "SOLVER_NOT_AVAILABLE",
        "objective_value": None,
        "example_output": {
            "production_plan": {},
            "ending_inventory": {},
            "ending_backlog": {}
        }
    }))
    raise SystemExit(0)

products = inp["products"]
quarters = inp["quarters"]
q_index = {q: i for i, q in enumerate(quarters)}
last_q = quarters[-1]

prod = {}
inv = {}
backlog = {}

for p in products:
    for q in quarters:
        prod[p, q] = solver.IntVar(0, solver.infinity(), f"prod_{p}_{q}")
        inv[p, q] = solver.IntVar(0, solver.infinity(), f"inv_{p}_{q}")
        backlog[p, q] = solver.IntVar(0, solver.infinity(), f"backlog_{p}_{q}")

for q in quarters:
    solver.Add(
        sum(inp["hours_per_unit"][p] * prod[p, q] for p in products)
        <= inp["production_hours_per_quarter"]
    )

for item in inp["production_forbidden"]:
    solver.Add(prod[item["product"], item["quarter"]] == 0)

for p in products:
    q = quarters[0]
    solver.Add(
        inv[p, q] - backlog[p, q]
        == inp["initial_inventory"][p] + prod[p, q] - inp["demand"][p][q_index[q]]
    )
    for prev_q, q in zip(quarters[:-1], quarters[1:]):
        solver.Add(
            inv[p, q] - backlog[p, q]
            == inv[p, prev_q] - backlog[p, prev_q] + prod[p, q] - inp["demand"][p][q_index[q]]
        )
    solver.Add(inv[p, last_q] == inp["required_ending_inventory"][p])
    solver.Add(backlog[p, last_q] == 0)

objective = solver.Objective()
for p in products:
    for q in quarters:
        objective.SetCoefficient(inv[p, q], inp["inventory_cost_per_unit_per_quarter"])
        objective.SetCoefficient(backlog[p, q], inp["delay_penalty_per_unit_per_quarter"][p])
objective.SetMinimization()

status_code = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "production_plan": {
            p: [int(round(prod[p, q].solution_value())) for q in quarters]
            for p in products
        },
        "ending_inventory": {
            p: [int(round(inv[p, q].solution_value())) for q in quarters]
            for p in products
        },
        "ending_backlog": {
            p: [int(round(backlog[p, q].solution_value())) for q in quarters]
            for p in products
        }
    }
    objective_value = solver.Objective().Value()
    if abs(objective_value - round(objective_value)) < 1e-9:
        objective_value = int(round(objective_value))
else:
    output = {
        "production_plan": {},
        "ending_inventory": {},
        "ending_backlog": {}
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status_code, str(status_code)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))