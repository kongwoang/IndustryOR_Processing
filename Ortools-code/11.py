import json
from ortools.linear_solver import pywraplp

inp = {
    "products": ["A", "B"],
    "processes": ["manufacturing", "assembly", "inspection"],
    "time_per_unit": {
        "A": {
            "manufacturing": 20,
            "assembly": 5,
            "inspection": 3
        },
        "B": {
            "manufacturing": 0,
            "assembly": 7,
            "inspection": 6
        }
    },
    "weekly_capacity": {
        "manufacturing": 120,
        "assembly": 80,
        "inspection": 40
    },
    "process_cost_per_hour": {
        "manufacturing": 12,
        "assembly": 8,
        "inspection": 10
    },
    "selling_price": {
        "A": 650,
        "B": 725
    },
    "decision_variable_type": "nonnegative integer",
    "goals": {
        "priority_1_min_profit": 3000,
        "priority_2_min_type_A_units": 5,
        "priority_3_minimize_weighted_idle_time": True
    }
}

def build_model():
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP solver is unavailable.")

    inf = solver.infinity()

    x_1 = solver.IntVar(0, inf, "x_1")
    x_2 = solver.IntVar(0, inf, "x_2")

    d1_minus = solver.NumVar(0.0, inf, "d1_minus")
    d1_plus = solver.NumVar(0.0, inf, "d1_plus")
    d2_minus = solver.NumVar(0.0, inf, "d2_minus")
    d2_plus = solver.NumVar(0.0, inf, "d2_plus")

    t = inp["time_per_unit"]
    cap = inp["weekly_capacity"]
    cost = inp["process_cost_per_hour"]
    price = inp["selling_price"]

    unit_profit_A = price["A"] - sum(t["A"][p] * cost[p] for p in inp["processes"])
    unit_profit_B = price["B"] - sum(t["B"][p] * cost[p] for p in inp["processes"])

    total_profit = solver.NumVar(-inf, inf, "total_profit")
    solver.Add(total_profit == unit_profit_A * x_1 + unit_profit_B * x_2)

    solver.Add(t["A"]["manufacturing"] * x_1 + t["B"]["manufacturing"] * x_2 <= cap["manufacturing"])
    solver.Add(t["A"]["assembly"] * x_1 + t["B"]["assembly"] * x_2 <= cap["assembly"])
    solver.Add(t["A"]["inspection"] * x_1 + t["B"]["inspection"] * x_2 <= cap["inspection"])

    solver.Add(total_profit + d1_minus - d1_plus == inp["goals"]["priority_1_min_profit"])
    solver.Add(x_1 + d2_minus - d2_plus == inp["goals"]["priority_2_min_type_A_units"])

    idle_m = solver.NumVar(0.0, inf, "idle_manufacturing")
    idle_a = solver.NumVar(0.0, inf, "idle_assembly")
    idle_i = solver.NumVar(0.0, inf, "idle_inspection")

    solver.Add(idle_m == cap["manufacturing"] - (t["A"]["manufacturing"] * x_1 + t["B"]["manufacturing"] * x_2))
    solver.Add(idle_a == cap["assembly"] - (t["A"]["assembly"] * x_1 + t["B"]["assembly"] * x_2))
    solver.Add(idle_i == cap["inspection"] - (t["A"]["inspection"] * x_1 + t["B"]["inspection"] * x_2))

    weighted_idle = solver.NumVar(0.0, inf, "weighted_idle")
    solver.Add(
        weighted_idle
        == cost["manufacturing"] * idle_m
        + cost["assembly"] * idle_a
        + cost["inspection"] * idle_i
    )

    return {
        "solver": solver,
        "x_1": x_1,
        "x_2": x_2,
        "d1_minus": d1_minus,
        "d1_plus": d1_plus,
        "d2_minus": d2_minus,
        "d2_plus": d2_plus,
        "idle_m": idle_m,
        "idle_a": idle_a,
        "idle_i": idle_i,
        "weighted_idle": weighted_idle,
        "total_profit": total_profit
    }

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
}

m = build_model()
solver = m["solver"]

obj1 = solver.Objective()
obj1.SetCoefficient(m["d1_minus"], 1.0)
obj1.SetMinimization()
status1 = solver.Solve()

if status1 not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "decision_variables": {"x_1": None, "x_2": None},
        "total_profit": None,
        "idle_time": {
            "manufacturing": None,
            "assembly": None,
            "inspection": None
        },
        "weighted_idle_time_cost": None,
        "goal_deviations": {
            "profit_shortfall": None,
            "type_A_shortfall": None
        }
    }
    print(json.dumps({
        "status": status_map.get(status1, str(status1)),
        "objective_value": None,
        "example_output": output
    }, ensure_ascii=False))
    raise SystemExit

best_d1_minus = m["d1_minus"].solution_value()
solver.Add(m["d1_minus"] == best_d1_minus)

obj2 = solver.Objective()
obj2.SetCoefficient(m["d2_minus"], 1.0)
obj2.SetMinimization()
status2 = solver.Solve()

if status2 not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "decision_variables": {"x_1": None, "x_2": None},
        "total_profit": None,
        "idle_time": {
            "manufacturing": None,
            "assembly": None,
            "inspection": None
        },
        "weighted_idle_time_cost": None,
        "goal_deviations": {
            "profit_shortfall": None,
            "type_A_shortfall": None
        }
    }
    print(json.dumps({
        "status": status_map.get(status2, str(status2)),
        "objective_value": None,
        "example_output": output
    }, ensure_ascii=False))
    raise SystemExit

best_d2_minus = m["d2_minus"].solution_value()
solver.Add(m["d2_minus"] == best_d2_minus)

obj3 = solver.Objective()
obj3.SetCoefficient(m["weighted_idle"], 1.0)
obj3.SetMinimization()
status3 = solver.Solve()

if status3 not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "decision_variables": {"x_1": None, "x_2": None},
        "total_profit": None,
        "idle_time": {
            "manufacturing": None,
            "assembly": None,
            "inspection": None
        },
        "weighted_idle_time_cost": None,
        "goal_deviations": {
            "profit_shortfall": None,
            "type_A_shortfall": None
        }
    }
    print(json.dumps({
        "status": status_map.get(status3, str(status3)),
        "objective_value": None,
        "example_output": output
    }, ensure_ascii=False))
    raise SystemExit

output = {
    "decision_variables": {
        "x_1": int(round(m["x_1"].solution_value())),
        "x_2": int(round(m["x_2"].solution_value()))
    },
    "total_profit": round(m["total_profit"].solution_value(), 6),
    "idle_time": {
        "manufacturing": round(m["idle_m"].solution_value(), 6),
        "assembly": round(m["idle_a"].solution_value(), 6),
        "inspection": round(m["idle_i"].solution_value(), 6)
    },
    "weighted_idle_time_cost": round(m["weighted_idle"].solution_value(), 6),
    "goal_deviations": {
        "profit_shortfall": round(m["d1_minus"].solution_value(), 6),
        "type_A_shortfall": round(m["d2_minus"].solution_value(), 6)
    }
}

print(json.dumps({
    "status": status_map.get(status3, str(status3)),
    "objective_value": round(m["total_profit"].solution_value(), 6),
    "example_output": output
}, ensure_ascii=False))