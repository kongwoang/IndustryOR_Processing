import json
from ortools.linear_solver import pywraplp

inp = {
    "initial_fund": 300000,
    "horizon_years": 3,
    "projects": [
        {
            "name": "project_1",
            "allowed_start_years": [1, 2, 3],
            "duration_years": 1,
            "return_multiplier": 1.2,
            "max_investment": None
        },
        {
            "name": "project_2",
            "allowed_start_years": [1],
            "duration_years": 2,
            "return_multiplier": 1.5,
            "max_investment": 150000
        },
        {
            "name": "project_3",
            "allowed_start_years": [2],
            "duration_years": 2,
            "return_multiplier": 1.6,
            "max_investment": 200000
        },
        {
            "name": "project_4",
            "allowed_start_years": [3],
            "duration_years": 1,
            "return_multiplier": 1.4,
            "max_investment": 100000
        }
    ]
}

projects = {p["name"]: p for p in inp["projects"]}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver.")

INF = solver.infinity()

# Decision variables
# Start of year 1
x11 = solver.NumVar(0.0, INF, "x11")  # project 1 in year 1
x21 = solver.NumVar(0.0, projects["project_2"]["max_investment"], "x21")  # project 2 in year 1
u1 = solver.NumVar(0.0, INF, "u1")    # uninvested cash carried to year 2

# Start of year 2
x12 = solver.NumVar(0.0, INF, "x12")  # project 1 in year 2
x32 = solver.NumVar(0.0, projects["project_3"]["max_investment"], "x32")  # project 3 in year 2
u2 = solver.NumVar(0.0, INF, "u2")    # uninvested cash carried to year 3

# Start of year 3
x13 = solver.NumVar(0.0, INF, "x13")  # project 1 in year 3
x43 = solver.NumVar(0.0, projects["project_4"]["max_investment"], "x43")  # project 4 in year 3
u3 = solver.NumVar(0.0, INF, "u3")    # uninvested cash until end of year 3

r1 = projects["project_1"]["return_multiplier"]
r2 = projects["project_2"]["return_multiplier"]
r3 = projects["project_3"]["return_multiplier"]
r4 = projects["project_4"]["return_multiplier"]

# Flow balance constraints
solver.Add(x11 + x21 + u1 == inp["initial_fund"])
solver.Add(x12 + x32 + u2 == r1 * x11 + u1)
solver.Add(x13 + x43 + u3 == r1 * x12 + r2 * x21 + u2)

# Objective: maximize total principal and interest at end of year 3
objective = solver.Objective()
objective.SetCoefficient(x13, r1)
objective.SetCoefficient(x32, r3)
objective.SetCoefficient(x43, r4)
objective.SetCoefficient(u3, 1.0)
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

def clean(v):
    if abs(v) < 1e-8:
        v = 0.0
    return round(v, 2)

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "investment_plan": {
            "year_1": {
                "project_1": clean(x11.solution_value()),
                "project_2": clean(x21.solution_value()),
                "uninvested_cash": clean(u1.solution_value())
            },
            "year_2": {
                "project_1": clean(x12.solution_value()),
                "project_3": clean(x32.solution_value()),
                "uninvested_cash": clean(u2.solution_value())
            },
            "year_3": {
                "project_1": clean(x13.solution_value()),
                "project_4": clean(x43.solution_value()),
                "uninvested_cash": clean(u3.solution_value())
            }
        },
        "ending_total_principal_and_interest": clean(objective.Value())
    }
    objective_value = clean(objective.Value())
else:
    output = {
        "investment_plan": {
            "year_1": {
                "project_1": None,
                "project_2": None,
                "uninvested_cash": None
            },
            "year_2": {
                "project_1": None,
                "project_3": None,
                "uninvested_cash": None
            },
            "year_3": {
                "project_1": None,
                "project_4": None,
                "uninvested_cash": None
            }
        },
        "ending_total_principal_and_interest": None
    }
    objective_value = None

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))