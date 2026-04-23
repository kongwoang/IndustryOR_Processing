import json
from ortools.linear_solver import pywraplp

inp = {
    "requirements": {
        "Mathematics": 2,
        "Operations Research": 2,
        "Computer Science": 2
    },
    "courses": [
        {
            "name": "Calculus",
            "categories": ["Mathematics"],
            "prerequisites": []
        },
        {
            "name": "Operations Research",
            "categories": ["Mathematics", "Operations Research"],
            "prerequisites": []
        },
        {
            "name": "Data Structures",
            "categories": ["Mathematics", "Computer Science"],
            "prerequisites": ["Computer Programming"]
        },
        {
            "name": "Management Statistics",
            "categories": ["Mathematics", "Operations Research"],
            "prerequisites": ["Calculus"]
        },
        {
            "name": "Computer Simulation",
            "categories": ["Computer Science", "Operations Research"],
            "prerequisites": ["Computer Programming"]
        },
        {
            "name": "Computer Programming",
            "categories": ["Computer Science"],
            "prerequisites": []
        },
        {
            "name": "Forecasting",
            "categories": ["Operations Research", "Mathematics"],
            "prerequisites": ["Management Statistics"]
        }
    ]
}

def build_solver(data, fixed_total=None, tie_break=False):
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP solver is not available.")

    courses = data["courses"]
    x = {course["name"]: solver.IntVar(0, 1, course["name"]) for course in courses}

    for category, req in data["requirements"].items():
        solver.Add(
            sum(x[course["name"]] for course in courses if category in course["categories"]) >= req
        )

    for course in courses:
        for prereq in course["prerequisites"]:
            solver.Add(x[course["name"]] <= x[prereq])

    total_courses = sum(x[course["name"]] for course in courses)
    if fixed_total is not None:
        solver.Add(total_courses == fixed_total)

    objective = solver.Objective()
    if tie_break:
        for idx, course in enumerate(courses):
            objective.SetCoefficient(x[course["name"]], 2 ** idx)
    else:
        for course in courses:
            objective.SetCoefficient(x[course["name"]], 1)
    objective.SetMinimization()

    return solver, x

status_name = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

solver1, x1 = build_solver(inp, fixed_total=None, tie_break=False)
status1 = solver1.Solve()

if status1 == pywraplp.Solver.OPTIMAL:
    min_courses = int(round(solver1.Objective().Value()))

    solver2, x2 = build_solver(inp, fixed_total=min_courses, tie_break=True)
    status2 = solver2.Solve()

    if status2 in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        selected_courses = [
            course["name"]
            for course in inp["courses"]
            if x2[course["name"]].solution_value() > 0.5
        ]
        output = {
            "minimum_number_of_courses": min_courses,
            "selected_courses": selected_courses
        }
        result = {
            "status": status_name.get(status2, str(status2)),
            "objective_value": min_courses,
            "example_output": output
        }
    else:
        output = {
            "minimum_number_of_courses": None,
            "selected_courses": []
        }
        result = {
            "status": status_name.get(status2, str(status2)),
            "objective_value": None,
            "example_output": output
        }
else:
    output = {
        "minimum_number_of_courses": None,
        "selected_courses": []
    }
    result = {
        "status": status_name.get(status1, str(status1)),
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))