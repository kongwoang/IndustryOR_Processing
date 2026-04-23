import json
import math
from ortools.linear_solver import pywraplp

inp = {
    "components": ["1", "2", "3"],
    "max_spares_per_component": 5,
    "reliability_by_component_and_spares": {
        "1": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "2": [0.6, 0.75, 0.95, 1.0, 1.0, 1.0],
        "3": [0.7, 0.9, 1.0, 1.0, 1.0, 1.0]
    },
    "unit_price_yuan": {
        "1": 20,
        "2": 30,
        "3": 40
    },
    "unit_weight_kg": {
        "1": 2,
        "2": 4,
        "3": 6
    },
    "budget_limit_yuan": 150,
    "weight_limit_kg": 20
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

components = inp["components"]
max_spares = inp["max_spares_per_component"]
rels = inp["reliability_by_component_and_spares"]
prices = inp["unit_price_yuan"]
weights = inp["unit_weight_kg"]

x = {}
for c in components:
    for k in range(max_spares + 1):
        x[(c, k)] = solver.BoolVar(f"x_{c}_{k}")

for c in components:
    solver.Add(sum(x[(c, k)] for k in range(max_spares + 1)) == 1)

solver.Add(
    sum(prices[c] * k * x[(c, k)] for c in components for k in range(max_spares + 1))
    <= inp["budget_limit_yuan"]
)
solver.Add(
    sum(weights[c] * k * x[(c, k)] for c in components for k in range(max_spares + 1))
    <= inp["weight_limit_kg"]
)

objective = solver.Objective()
for c in components:
    for k in range(max_spares + 1):
        objective.SetCoefficient(x[(c, k)], math.log(rels[c][k]))
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

output = {
    "selected_spares": {},
    "component_reliabilities": {},
    "total_cost_yuan": None,
    "total_weight_kg": None,
    "system_reliability": None
}

objective_value = None

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    total_cost = 0
    total_weight = 0
    system_reliability = 1.0

    for c in components:
        chosen_k = None
        for k in range(max_spares + 1):
            if x[(c, k)].solution_value() > 0.5:
                chosen_k = k
                break
        if chosen_k is None:
            raise RuntimeError(f"No spare count selected for component {c}.")

        r = rels[c][chosen_k]
        output["selected_spares"][c] = chosen_k
        output["component_reliabilities"][c] = round(r, 10)

        total_cost += prices[c] * chosen_k
        total_weight += weights[c] * chosen_k
        system_reliability *= r

    output["total_cost_yuan"] = total_cost
    output["total_weight_kg"] = total_weight
    output["system_reliability"] = round(system_reliability, 10)
    objective_value = round(system_reliability, 10)

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))