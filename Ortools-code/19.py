# Source: :contentReference[oaicite:0]{index=0}
import json
from ortools.linear_solver import pywraplp

inp = {
    "cost_per_chair": {
        "A": 50,
        "B": 45,
        "C": 40
    },
    "chairs_per_order": {
        "A": 15,
        "B": 10,
        "C": 10
    },
    "min_total_chairs": 100,
    "max_total_chairs": 500,
    "logic": {
        "if_A_then_at_least_one_B_order": True,
        "if_B_then_at_least_one_C_order": True
    }
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

manufacturers = ["A", "B", "C"]

max_orders = {
    "A": inp["max_total_chairs"] // inp["chairs_per_order"]["A"],
    "B": inp["max_total_chairs"] // inp["chairs_per_order"]["B"],
    "C": inp["max_total_chairs"] // inp["chairs_per_order"]["C"],
}

x = {
    m: solver.IntVar(0, max_orders[m], f"x_{m}")
    for m in manufacturers
}

yA = solver.BoolVar("yA_used")
yB = solver.BoolVar("yB_used")

total_chairs_expr = solver.Sum(
    inp["chairs_per_order"][m] * x[m] for m in manufacturers
)

total_cost_expr = solver.Sum(
    inp["cost_per_chair"][m] * inp["chairs_per_order"][m] * x[m]
    for m in manufacturers
)

solver.Add(total_chairs_expr >= inp["min_total_chairs"])
solver.Add(total_chairs_expr <= inp["max_total_chairs"])

solver.Add(x["A"] <= max_orders["A"] * yA)
solver.Add(x["B"] >= yA)

solver.Add(x["B"] <= max_orders["B"] * yB)
solver.Add(x["C"] >= yB)

solver.Minimize(total_cost_expr)

status = solver.Solve()

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "orders": {
            "A": int(round(x["A"].solution_value())),
            "B": int(round(x["B"].solution_value())),
            "C": int(round(x["C"].solution_value()))
        },
        "total_chairs": int(round(sum(
            inp["chairs_per_order"][m] * x[m].solution_value() for m in manufacturers
        ))),
        "total_cost": float(sum(
            inp["cost_per_chair"][m] * inp["chairs_per_order"][m] * x[m].solution_value()
            for m in manufacturers
        ))
    }
    result = {
        "status": "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE",
        "objective_value": float(solver.Objective().Value()),
        "example_output": output
    }
else:
    output = {
        "orders": {"A": 0, "B": 0, "C": 0},
        "total_chairs": 0,
        "total_cost": 0.0
    }
    result = {
        "status": "INFEASIBLE" if status == pywraplp.Solver.INFEASIBLE else "NO_SOLUTION",
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))