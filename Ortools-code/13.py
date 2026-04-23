import json
from ortools.linear_solver import pywraplp

inp = {
    "total_acres": 100,
    "profit_per_acre": {
        "corn": 1500,
        "wheat": 1200,
        "soybeans": 1800,
        "sorghum": 1600
    },
    "constraints": {
        "corn_min_times_wheat": 2,
        "soybeans_min_fraction_of_sorghum": 0.5,
        "wheat_equals_times_sorghum": 3
    }
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver")

corn = solver.NumVar(0.0, solver.infinity(), "corn_acres")
wheat = solver.NumVar(0.0, solver.infinity(), "wheat_acres")
soybeans = solver.NumVar(0.0, solver.infinity(), "soybeans_acres")
sorghum = solver.NumVar(0.0, solver.infinity(), "sorghum_acres")

# Total land limit
solver.Add(corn + wheat + soybeans + sorghum <= inp["total_acres"])

# Problem constraints
solver.Add(corn >= inp["constraints"]["corn_min_times_wheat"] * wheat)
solver.Add(soybeans >= inp["constraints"]["soybeans_min_fraction_of_sorghum"] * sorghum)
solver.Add(wheat == inp["constraints"]["wheat_equals_times_sorghum"] * sorghum)

# Objective: maximize profit
objective = solver.Objective()
objective.SetCoefficient(corn, inp["profit_per_acre"]["corn"])
objective.SetCoefficient(wheat, inp["profit_per_acre"]["wheat"])
objective.SetCoefficient(soybeans, inp["profit_per_acre"]["soybeans"])
objective.SetCoefficient(sorghum, inp["profit_per_acre"]["sorghum"])
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
    output = {
        "corn_acres": corn.solution_value(),
        "wheat_acres": wheat.solution_value(),
        "soybeans_acres": soybeans.solution_value(),
        "sorghum_acres": sorghum.solution_value()
    }
    objective_value = objective.Value()
else:
    output = {
        "corn_acres": None,
        "wheat_acres": None,
        "soybeans_acres": None,
        "sorghum_acres": None
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))