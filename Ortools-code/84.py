import json
from ortools.linear_solver import pywraplp

inp = {
    "candidates": [
        {
            "name": "A",
            "salary": 8100,
            "degree": "Bachelor's",
            "experience": 3
        },
        {
            "name": "B",
            "salary": 20000,
            "degree": "Master's",
            "experience": 10
        },
        {
            "name": "C",
            "salary": 21000,
            "degree": "Doctoral",
            "experience": 4
        },
        {
            "name": "D",
            "salary": 3000,
            "degree": "None",
            "experience": 3
        },
        {
            "name": "E",
            "salary": 8000,
            "degree": "None",
            "experience": 7
        }
    ],
    "constraints": {
        "min_hires": 2,
        "max_hires": 3,
        "budget": 35000,
        "min_total_experience": 12,
        "require_at_least_one_master_or_doctoral": True,
        "mutually_exclusive_pairs": [
            ["A", "E"]
        ]
    }
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

candidates = inp["candidates"]
constraints = inp["constraints"]
names = [c["name"] for c in candidates]

x = {c["name"]: solver.BoolVar(f"x_{c['name']}") for c in candidates}

solver.Minimize(solver.Sum(c["salary"] * x[c["name"]] for c in candidates))

solver.Add(solver.Sum(x[name] for name in names) >= constraints["min_hires"])
solver.Add(solver.Sum(x[name] for name in names) <= constraints["max_hires"])
solver.Add(solver.Sum(c["salary"] * x[c["name"]] for c in candidates) <= constraints["budget"])
solver.Add(solver.Sum(c["experience"] * x[c["name"]] for c in candidates) >= constraints["min_total_experience"])

if constraints["require_at_least_one_master_or_doctoral"]:
    solver.Add(
        solver.Sum(
            x[c["name"]]
            for c in candidates
            if c["degree"] in {"Master's", "Doctoral"}
        ) >= 1
    )

for a, b in constraints["mutually_exclusive_pairs"]:
    solver.Add(x[a] + x[b] <= 1)

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}
status_str = status_map.get(status, str(status))

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    hire_decisions = {name: int(round(x[name].solution_value())) for name in names}
    selected_candidates = [name for name in names if hire_decisions[name] == 1]
    total_salary = sum(c["salary"] * hire_decisions[c["name"]] for c in candidates)
    total_experience = sum(c["experience"] * hire_decisions[c["name"]] for c in candidates)
    output = {
        "selected_candidates": selected_candidates,
        "hire_decisions": hire_decisions,
        "num_hired": len(selected_candidates),
        "total_salary": total_salary,
        "total_experience": total_experience
    }
    objective_value = int(round(solver.Objective().Value()))
else:
    output = {
        "selected_candidates": [],
        "hire_decisions": {name: 0 for name in names},
        "num_hired": 0,
        "total_salary": 0,
        "total_experience": 0
    }
    objective_value = None

print(json.dumps({
    "status": status_str,
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))