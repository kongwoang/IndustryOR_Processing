import json
from ortools.linear_solver import pywraplp

inp = {
    "employee_types": [
        {
            "name": "full_time",
            "count": 5,
            "regular_hours_per_employee": 160,
            "sales_pairs_per_hour": 5,
            "regular_wage_yuan_per_hour": 1.0,
            "overtime_wage_yuan_per_hour": 1.5
        },
        {
            "name": "part_time",
            "count": 4,
            "regular_hours_per_employee": 80,
            "sales_pairs_per_hour": 2,
            "regular_wage_yuan_per_hour": 0.6,
            "overtime_wage_yuan_per_hour": 0.7
        }
    ],
    "profit_per_pair_yuan": 0.3,
    "goals": {
        "sales_target_pairs": 5500,
        "priorities": [
            "meet_sales_target",
            "fully_use_regular_hours",
            "minimize_overtime_hours"
        ]
    }
}


def clean(x):
    if abs(x - round(x)) <= 1e-8:
        return int(round(x))
    return round(x, 6)


def status_to_string(code):
    mapping = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    return mapping.get(code, str(code))


def build_model(data, sales_dev_cap=None, unused_hours_cap=None):
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("Failed to create GLOP solver")

    ft = next(e for e in data["employee_types"] if e["name"] == "full_time")
    pt = next(e for e in data["employee_types"] if e["name"] == "part_time")

    ft_cap = ft["count"] * ft["regular_hours_per_employee"]
    pt_cap = pt["count"] * pt["regular_hours_per_employee"]

    rf = solver.NumVar(0.0, ft_cap, "rf")
    rp = solver.NumVar(0.0, pt_cap, "rp")
    of = solver.NumVar(0.0, solver.infinity(), "of")
    op = solver.NumVar(0.0, solver.infinity(), "op")

    d1_minus = solver.NumVar(0.0, solver.infinity(), "d1_minus")
    d1_plus = solver.NumVar(0.0, solver.infinity(), "d1_plus")
    d2_full = solver.NumVar(0.0, solver.infinity(), "d2_full")
    d2_part = solver.NumVar(0.0, solver.infinity(), "d2_part")

    solver.Add(
        ft["sales_pairs_per_hour"] * (rf + of) +
        pt["sales_pairs_per_hour"] * (rp + op) +
        d1_minus - d1_plus ==
        data["goals"]["sales_target_pairs"]
    )

    solver.Add(rf + d2_full == ft_cap)
    solver.Add(rp + d2_part == pt_cap)

    if sales_dev_cap is not None:
        solver.Add(d1_minus + d1_plus <= sales_dev_cap + 1e-7)

    if unused_hours_cap is not None:
        solver.Add(d2_full + d2_part <= unused_hours_cap + 1e-7)

    return solver, {
        "rf": rf,
        "rp": rp,
        "of": of,
        "op": op,
        "d1_minus": d1_minus,
        "d1_plus": d1_plus,
        "d2_full": d2_full,
        "d2_part": d2_part
    }


def set_objective_minimize(solver, terms):
    obj = solver.Objective()
    for var, coef in terms:
        obj.SetCoefficient(var, coef)
    obj.SetMinimization()


def solve_optimal(solver):
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(status_to_string(status))
    return status


try:
    solver1, v1 = build_model(inp)
    set_objective_minimize(solver1, [(v1["d1_minus"], 1.0), (v1["d1_plus"], 1.0)])
    solve_optimal(solver1)
    best_sales_dev = v1["d1_minus"].solution_value() + v1["d1_plus"].solution_value()

    solver2, v2 = build_model(inp, sales_dev_cap=best_sales_dev)
    set_objective_minimize(solver2, [(v2["d2_full"], 1.0), (v2["d2_part"], 1.0)])
    solve_optimal(solver2)
    best_unused_hours = v2["d2_full"].solution_value() + v2["d2_part"].solution_value()

    solver3, v3 = build_model(inp, sales_dev_cap=best_sales_dev, unused_hours_cap=best_unused_hours)
    set_objective_minimize(solver3, [(v3["of"], 1.0), (v3["op"], 1.0)])
    status = solve_optimal(solver3)

    ft = next(e for e in inp["employee_types"] if e["name"] == "full_time")
    pt = next(e for e in inp["employee_types"] if e["name"] == "part_time")

    rf = v3["rf"].solution_value()
    rp = v3["rp"].solution_value()
    of = v3["of"].solution_value()
    op = v3["op"].solution_value()
    d1m = v3["d1_minus"].solution_value()
    d1p = v3["d1_plus"].solution_value()
    d2f = v3["d2_full"].solution_value()
    d2p = v3["d2_part"].solution_value()

    sales_pairs = ft["sales_pairs_per_hour"] * (rf + of) + pt["sales_pairs_per_hour"] * (rp + op)
    sales_profit = sales_pairs * inp["profit_per_pair_yuan"]
    labor_cost = (
        rf * ft["regular_wage_yuan_per_hour"] +
        rp * pt["regular_wage_yuan_per_hour"] +
        of * ft["overtime_wage_yuan_per_hour"] +
        op * pt["overtime_wage_yuan_per_hour"]
    )

    output = {
        "regular_hours_used": {
            "full_time": clean(rf),
            "part_time": clean(rp)
        },
        "overtime_hours": {
            "full_time": clean(of),
            "part_time": clean(op)
        },
        "sales_pairs": clean(sales_pairs),
        "sales_profit_yuan": clean(sales_profit),
        "labor_cost_yuan": clean(labor_cost)
        ,
        "goal_deviations": {
            "sales_underachievement_pairs": clean(d1m),
            "sales_overachievement_pairs": clean(d1p),
            "full_time_unused_regular_hours": clean(d2f),
            "part_time_unused_regular_hours": clean(d2p)
        }
    }

    result = {
        "status": status_to_string(status),
        "objective_value": clean(of + op),
        "example_output": output
    }
except Exception as e:
    output = {
        "regular_hours_used": {
            "full_time": None,
            "part_time": None
        },
        "overtime_hours": {
            "full_time": None,
            "part_time": None
        },
        "sales_pairs": None,
        "sales_profit_yuan": None,
        "labor_cost_yuan": None,
        "goal_deviations": {
            "sales_underachievement_pairs": None,
            "sales_overachievement_pairs": None,
            "full_time_unused_regular_hours": None,
            "part_time_unused_regular_hours": None
        }
    }
    result = {
        "status": str(e),
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))