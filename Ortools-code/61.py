# Source problem statement: :contentReference[oaicite:0]{index=0}
import json
from ortools.linear_solver import pywraplp

inp = {
    "devices": [
        {
            "name": "A",
            "prep_completion_cost": 1000,
            "unit_production_cost": 20,
            "max_processing_capacity": 900
        },
        {
            "name": "B",
            "prep_completion_cost": 920,
            "unit_production_cost": 24,
            "max_processing_capacity": 1000
        },
        {
            "name": "C",
            "prep_completion_cost": 800,
            "unit_production_cost": 16,
            "max_processing_capacity": 1200
        },
        {
            "name": "D",
            "prep_completion_cost": 700,
            "unit_production_cost": 28,
            "max_processing_capacity": 1600
        }
    ],
    "required_production": 2000
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("Failed to create SCIP solver.")

devices = inp["devices"]
required = inp["required_production"]

x = {}
y = {}

for d in devices:
    name = d["name"]
    x[name] = solver.IntVar(0, int(d["max_processing_capacity"]), f"x_{name}")
    y[name] = solver.BoolVar(f"y_{name}")
    solver.Add(x[name] <= d["max_processing_capacity"] * y[name])

solver.Add(solver.Sum(x[d["name"]] for d in devices) == required)

objective = solver.Objective()
for d in devices:
    name = d["name"]
    objective.SetCoefficient(y[name], d["prep_completion_cost"])
    objective.SetCoefficient(x[name], d["unit_production_cost"])
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
    production_plan = {d["name"]: int(round(x[d["name"]].solution_value())) for d in devices}
    enabled_devices = [d["name"] for d in devices if y[d["name"]].solution_value() > 0.5]
    total_cost = float(objective.Value())
else:
    production_plan = {d["name"]: 0 for d in devices}
    enabled_devices = []
    total_cost = None

output = {
    "enabled_devices": enabled_devices,
    "production_plan": production_plan,
    "total_cost": total_cost
}

result = {
    "status": status_map.get(status_code, str(status_code)),
    "objective_value": total_cost,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))