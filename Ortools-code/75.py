import json
from ortools.linear_solver import pywraplp

inp = {
    "products": {
        "A": {
            "process_1_time": 2,
            "process_2_time": 3,
            "unit_profit": 4
        },
        "B": {
            "process_1_time": 3,
            "process_2_time": 4,
            "unit_profit": 10,
            "byproduct_C_generated": 2
        }
    },
    "processes": {
        "process_1_capacity": 16,
        "process_2_capacity": 24
    },
    "byproduct_C": {
        "sale_limit": 5,
        "sale_profit": 3,
        "disposal_cost": 2
    }
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver.")

A = solver.NumVar(0.0, solver.infinity(), "A_units")
B = solver.NumVar(0.0, solver.infinity(), "B_units")
C_sold = solver.NumVar(0.0, solver.infinity(), "C_sold_units")
C_disposed = solver.NumVar(0.0, solver.infinity(), "C_disposed_units")

solver.Add(
    inp["products"]["A"]["process_1_time"] * A +
    inp["products"]["B"]["process_1_time"] * B
    <= inp["processes"]["process_1_capacity"]
)
solver.Add(
    inp["products"]["A"]["process_2_time"] * A +
    inp["products"]["B"]["process_2_time"] * B
    <= inp["processes"]["process_2_capacity"]
)
solver.Add(
    C_sold + C_disposed ==
    inp["products"]["B"]["byproduct_C_generated"] * B
)
solver.Add(C_sold <= inp["byproduct_C"]["sale_limit"])

objective = solver.Objective()
objective.SetCoefficient(A, inp["products"]["A"]["unit_profit"])
objective.SetCoefficient(B, inp["products"]["B"]["unit_profit"])
objective.SetCoefficient(C_sold, inp["byproduct_C"]["sale_profit"])
objective.SetCoefficient(C_disposed, -inp["byproduct_C"]["disposal_cost"])
objective.SetMaximization()

status_code = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}
status = status_map.get(status_code, "UNKNOWN")

if status in {"OPTIMAL", "FEASIBLE"}:
    output = {
        "decision_variables": {
            "A_units": round(A.solution_value(), 6),
            "B_units": round(B.solution_value(), 6),
            "C_sold_units": round(C_sold.solution_value(), 6),
            "C_disposed_units": round(C_disposed.solution_value(), 6)
        },
        "total_profit": round(objective.Value(), 6)
    }
    objective_value = round(objective.Value(), 6)
else:
    output = {
        "decision_variables": {
            "A_units": None,
            "B_units": None,
            "C_sold_units": None,
            "C_disposed_units": None
        },
        "total_profit": None
    }
    objective_value = None

print(json.dumps({
    "status": status,
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))