import json
from ortools.linear_solver import pywraplp

inp = {
    "years": [1, 2],
    "production_by_year": {
        "1": 10,
        "2": 15
    },
    "pilots_trained_per_training_jet_per_year": 5
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("Failed to create solver.")

years = inp["years"]
production_by_year = inp["production_by_year"]
pilots_per_training_jet = inp["pilots_trained_per_training_jet_per_year"]

training_jets = {}
for y in years:
    key = str(y)
    training_jets[key] = solver.IntVar(0, production_by_year[key], f"training_jets_{key}")

solver.Maximize(
    solver.Sum(pilots_per_training_jet * training_jets[str(y)] for y in years)
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
    output = {
        "training_jets_by_year": {
            str(y): int(round(training_jets[str(y)].solution_value())) for y in years
        },
        "total_trained_pilots_by_end_of_year_2": int(round(solver.Objective().Value()))
    }
    objective_value = int(round(solver.Objective().Value()))
else:
    output = {
        "training_jets_by_year": {
            str(y): None for y in years
        },
        "total_trained_pilots_by_end_of_year_2": None
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))