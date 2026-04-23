import json
from ortools.linear_solver import pywraplp

inp = {
    "candidates": [
        {
            "name": "F",
            "salary": 12000,
            "skill_level": 2,
            "experience_years": 1
        },
        {
            "name": "G",
            "salary": 15000,
            "skill_level": 3,
            "experience_years": 2
        },
        {
            "name": "H",
            "salary": 18000,
            "skill_level": 4,
            "experience_years": 2
        },
        {
            "name": "I",
            "salary": 5000,
            "skill_level": 1,
            "experience_years": 5
        },
        {
            "name": "J",
            "salary": 10000,
            "skill_level": 2,
            "experience_years": 4
        }
    ],
    "budget": 40000,
    "max_hires": 4,
    "min_total_skill": 8,
    "min_total_experience": 8,
    "mutually_exclusive_pairs": [
        ["G", "J"]
    ]
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

candidates = inp["candidates"]
names = [c["name"] for c in candidates]
salary = {c["name"]: c["salary"] for c in candidates}
skill = {c["name"]: c["skill_level"] for c in candidates}
experience = {c["name"]: c["experience_years"] for c in candidates}

x = {name: solver.BoolVar(f"x_{name}") for name in names}

solver.Add(sum(salary[name] * x[name] for name in names) <= inp["budget"])
solver.Add(sum(x[name] for name in names) <= inp["max_hires"])
solver.Add(sum(skill[name] * x[name] for name in names) >= inp["min_total_skill"])
solver.Add(sum(experience[name] * x[name] for name in names) >= inp["min_total_experience"])

for a, b in inp["mutually_exclusive_pairs"]:
    solver.Add(x[a] + x[b] <= 1)

objective = solver.Objective()
for name in names:
    objective.SetCoefficient(x[name], salary[name])
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
    selected_candidates = [name for name in names if x[name].solution_value() > 0.5]
    total_salary = sum(salary[name] for name in selected_candidates)
    total_skill = sum(skill[name] for name in selected_candidates)
    total_experience = sum(experience[name] for name in selected_candidates)
    output = {
        "selected_candidates": selected_candidates,
        "total_salary": total_salary,
        "total_skill": total_skill,
        "total_experience": total_experience
    }
    objective_value = solver.Objective().Value()
else:
    output = {
        "selected_candidates": [],
        "total_salary": None,
        "total_skill": None,
        "total_experience": None
    }
    objective_value = None

result = {
    "status": status_map.get(status, "UNKNOWN"),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))