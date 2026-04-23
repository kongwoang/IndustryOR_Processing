# Source: :contentReference[oaicite:0]{index=0}
import json
from ortools.linear_solver import pywraplp

inp = {
    "profits": {
        "trucks": 5,
        "airplanes": 10,
        "boats": 8,
        "trains": 7
    },
    "resources": {
        "wood_available": 890,
        "steel_available": 500
    },
    "resource_requirements": {
        "wood": {
            "trucks": 12,
            "airplanes": 20,
            "boats": 15,
            "trains": 10
        },
        "steel": {
            "trucks": 6,
            "airplanes": 3,
            "boats": 5,
            "trains": 4
        }
    },
    "logic_constraints": {
        "if_trucks_then_no_trains": True,
        "if_boats_then_also_airplanes": True,
        "boats_cannot_exceed_trains": True
    }
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    solver = pywraplp.Solver.CreateSolver("CBC")
if solver is None:
    raise RuntimeError("No suitable MIP solver is available in OR-Tools.")

profits = inp["profits"]
wood_available = inp["resources"]["wood_available"]
steel_available = inp["resources"]["steel_available"]
wood_req = inp["resource_requirements"]["wood"]
steel_req = inp["resource_requirements"]["steel"]

toys = ["trucks", "airplanes", "boats", "trains"]

upper_bounds = {}
for toy in toys:
    ub_wood = wood_available // wood_req[toy]
    ub_steel = steel_available // steel_req[toy]
    upper_bounds[toy] = min(ub_wood, ub_steel)

x = {toy: solver.IntVar(0, upper_bounds[toy], f"x_{toy}") for toy in toys}
y = {toy: solver.BoolVar(f"y_{toy}") for toy in toys}

for toy in toys:
    solver.Add(x[toy] <= upper_bounds[toy] * y[toy])
    solver.Add(x[toy] >= y[toy])

solver.Add(sum(wood_req[toy] * x[toy] for toy in toys) <= wood_available)
solver.Add(sum(steel_req[toy] * x[toy] for toy in toys) <= steel_available)

solver.Add(y["trucks"] + y["trains"] <= 1)
solver.Add(y["boats"] <= y["airplanes"])
solver.Add(x["boats"] <= x["trains"])

solver.Maximize(sum(profits[toy] * x[toy] for toy in toys))

status_code = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
}

if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "production_quantities": {
            "trucks": int(round(x["trucks"].solution_value())),
            "airplanes": int(round(x["airplanes"].solution_value())),
            "boats": int(round(x["boats"].solution_value())),
            "trains": int(round(x["trains"].solution_value()))
        }
    }
    objective_value = int(round(solver.Objective().Value()))
else:
    output = {
        "production_quantities": {
            "trucks": 0,
            "airplanes": 0,
            "boats": 0,
            "trains": 0
        }
    }
    objective_value = None

result = {
    "status": status_map.get(status_code, "UNKNOWN"),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, separators=(",", ":")))