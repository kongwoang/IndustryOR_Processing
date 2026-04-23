import json
from ortools.linear_solver import pywraplp

inp = {
    "cities": [
        "Donghai City",
        "Nanjiang City"
    ],
    "specialties": [
        1,
        2,
        3
    ],
    "demands": [
        {
            "city": "Donghai City",
            "specialty": 1,
            "demand": 1000
        },
        {
            "city": "Donghai City",
            "specialty": 2,
            "demand": 2000
        },
        {
            "city": "Donghai City",
            "specialty": 3,
            "demand": 1500
        },
        {
            "city": "Nanjiang City",
            "specialty": 1,
            "demand": 2000
        },
        {
            "city": "Nanjiang City",
            "specialty": 2,
            "demand": 1000
        },
        {
            "city": "Nanjiang City",
            "specialty": 3,
            "demand": 1000
        }
    ],
    "applicant_types": [
        {
            "type": 1,
            "available": 1500,
            "suitable_specialties": [
                1,
                2
            ],
            "preferred_specialty": 1,
            "preferred_city": "Donghai City"
        },
        {
            "type": 2,
            "available": 1500,
            "suitable_specialties": [
                2,
                3
            ],
            "preferred_specialty": 2,
            "preferred_city": "Donghai City"
        },
        {
            "type": 3,
            "available": 1500,
            "suitable_specialties": [
                1,
                3
            ],
            "preferred_specialty": 1,
            "preferred_city": "Nanjiang City"
        },
        {
            "type": 4,
            "available": 1500,
            "suitable_specialties": [
                1,
                3
            ],
            "preferred_specialty": 3,
            "preferred_city": "Nanjiang City"
        },
        {
            "type": 5,
            "available": 1500,
            "suitable_specialties": [
                2,
                3
            ],
            "preferred_specialty": 3,
            "preferred_city": "Donghai City"
        },
        {
            "type": 6,
            "available": 1500,
            "suitable_specialties": [
                3
            ],
            "preferred_specialty": 3,
            "preferred_city": "Nanjiang City"
        }
    ],
    "priority_goals": {
        "p1": "Fully satisfy all branch-specialty demands",
        "p2_preferred_specialty_target": 8000,
        "p3_preferred_city_target": 8000
    }
}


def build_model():
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP solver is not available.")

    cities = inp["cities"]
    applicant_types = inp["applicant_types"]
    demands = {(d["city"], d["specialty"]): d["demand"] for d in inp["demands"]}

    x = {}
    for t in applicant_types:
        i = t["type"]
        for city in cities:
            for sp in t["suitable_specialties"]:
                x[(i, city, sp)] = solver.IntVar(0, t["available"], f"x_{i}_{city}_{sp}")

    for (city, sp), demand in demands.items():
        solver.Add(
            sum(x[(t["type"], city, sp)] for t in applicant_types if (t["type"], city, sp) in x) == demand
        )

    for t in applicant_types:
        i = t["type"]
        solver.Add(
            sum(
                x[(i, city, sp)]
                for city in cities
                for sp in t["suitable_specialties"]
                if (i, city, sp) in x
            ) <= t["available"]
        )

    preferred_specialty_expr = solver.Sum(
        x[(t["type"], city, t["preferred_specialty"])]
        for t in applicant_types
        for city in cities
        if (t["type"], city, t["preferred_specialty"]) in x
    )

    preferred_city_expr = solver.Sum(
        x[(t["type"], t["preferred_city"], sp)]
        for t in applicant_types
        for sp in t["suitable_specialties"]
        if (t["type"], t["preferred_city"], sp) in x
    )

    total_recruited_expr = solver.Sum(x.values())

    return solver, x, preferred_specialty_expr, preferred_city_expr, total_recruited_expr


def solve_stage_1():
    solver, x, preferred_specialty_expr, preferred_city_expr, total_recruited_expr = build_model()
    solver.Minimize(0)
    status = solver.Solve()
    return status, solver, x, preferred_specialty_expr, preferred_city_expr, total_recruited_expr


def solve_stage_2():
    solver, x, preferred_specialty_expr, preferred_city_expr, total_recruited_expr = build_model()
    solver.Maximize(preferred_specialty_expr)
    status = solver.Solve()
    best_preferred_specialty = int(round(preferred_specialty_expr.solution_value())) if status in (
        pywraplp.Solver.OPTIMAL,
        pywraplp.Solver.FEASIBLE,
    ) else None
    return status, best_preferred_specialty


def solve_stage_3(best_preferred_specialty):
    solver, x, preferred_specialty_expr, preferred_city_expr, total_recruited_expr = build_model()
    solver.Add(preferred_specialty_expr == best_preferred_specialty)
    solver.Maximize(preferred_city_expr)
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return status, None, None
    best_preferred_city = int(round(preferred_city_expr.solution_value()))
    total_recruited = int(round(total_recruited_expr.solution_value()))
    return status, best_preferred_city, total_recruited


stage1_status, _, _, _, _, _ = solve_stage_1()
if stage1_status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    print(json.dumps({
        "status": "infeasible_at_p1",
        "objective_value": None,
        "example_output": None
    }, ensure_ascii=False))
    raise SystemExit

stage2_status, best_preferred_specialty = solve_stage_2()
if stage2_status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    print(json.dumps({
        "status": "infeasible_at_p2",
        "objective_value": None,
        "example_output": None
    }, ensure_ascii=False))
    raise SystemExit

stage3_status, best_preferred_city, total_recruited = solve_stage_3(best_preferred_specialty)
if stage3_status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    print(json.dumps({
        "status": "infeasible_at_p3",
        "objective_value": None,
        "example_output": None
    }, ensure_ascii=False))
    raise SystemExit

p2_target = inp["priority_goals"]["p2_preferred_specialty_target"]
p3_target = inp["priority_goals"]["p3_preferred_city_target"]

output = {
    "total_recruited": total_recruited,
    "max_preferred_specialty_people_under_p1": best_preferred_specialty,
    "p2_shortfall": max(0, p2_target - best_preferred_specialty),
    "max_preferred_city_people_under_p1_and_best_p2": best_preferred_city,
    "p3_shortfall": max(0, p3_target - best_preferred_city),
    "recruited_people_not_in_preferred_city": total_recruited - best_preferred_city
}

result = {
    "status": "optimal",
    "objective_value": output["p3_shortfall"],
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))