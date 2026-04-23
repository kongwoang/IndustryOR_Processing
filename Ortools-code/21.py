import json
from ortools.linear_solver import pywraplp

inp = {
    "suppliers": {
        "A": {
            "cost_per_table": 120,
            "tables_per_order": 20
        },
        "B": {
            "cost_per_table": 110,
            "tables_per_order": 15
        },
        "C": {
            "cost_per_table": 100,
            "tables_per_order": 15
        }
    },
    "requirements": {
        "min_total_tables": 150,
        "max_total_tables": 600
    },
    "logic_constraints": {
        "if_order_A_then_min_tables_from_B": 30,
        "if_order_B_then_order_C": True
    }
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

suppliers = inp["suppliers"]
req = inp["requirements"]
logic = inp["logic_constraints"]

max_orders = {
    s: req["max_total_tables"] // suppliers[s]["tables_per_order"]
    for s in suppliers
}

xA = solver.IntVar(0, max_orders["A"], "xA")
xB = solver.IntVar(0, max_orders["B"], "xB")
xC = solver.IntVar(0, max_orders["C"], "xC")

yA = solver.IntVar(0, 1, "yA")
yB = solver.IntVar(0, 1, "yB")

tables_A = suppliers["A"]["tables_per_order"] * xA
tables_B = suppliers["B"]["tables_per_order"] * xB
tables_C = suppliers["C"]["tables_per_order"] * xC
total_tables = tables_A + tables_B + tables_C

solver.Add(total_tables >= req["min_total_tables"])
solver.Add(total_tables <= req["max_total_tables"])

solver.Add(xA <= max_orders["A"] * yA)
solver.Add(xB <= max_orders["B"] * yB)

min_orders_B_if_A = (
    logic["if_order_A_then_min_tables_from_B"] + suppliers["B"]["tables_per_order"] - 1
) // suppliers["B"]["tables_per_order"]
solver.Add(xB >= min_orders_B_if_A * yA)

if logic["if_order_B_then_order_C"]:
    solver.Add(xC >= yB)

objective = solver.Objective()
objective.SetCoefficient(xA, suppliers["A"]["cost_per_table"] * suppliers["A"]["tables_per_order"])
objective.SetCoefficient(xB, suppliers["B"]["cost_per_table"] * suppliers["B"]["tables_per_order"])
objective.SetCoefficient(xC, suppliers["C"]["cost_per_table"] * suppliers["C"]["tables_per_order"])
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

def clean_number(v):
    if abs(v - round(v)) <= 1e-9:
        return int(round(v))
    return float(v)

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "orders": {
            "A": clean_number(xA.solution_value()),
            "B": clean_number(xB.solution_value()),
            "C": clean_number(xC.solution_value())
        },
        "tables": {
            "A": clean_number(suppliers["A"]["tables_per_order"] * xA.solution_value()),
            "B": clean_number(suppliers["B"]["tables_per_order"] * xB.solution_value()),
            "C": clean_number(suppliers["C"]["tables_per_order"] * xC.solution_value()),
            "total": clean_number(
                suppliers["A"]["tables_per_order"] * xA.solution_value()
                + suppliers["B"]["tables_per_order"] * xB.solution_value()
                + suppliers["C"]["tables_per_order"] * xC.solution_value()
            )
        },
        "total_cost": clean_number(objective.Value())
    }
    objective_value = clean_number(objective.Value())
else:
    output = {
        "orders": {
            "A": 0,
            "B": 0,
            "C": 0
        },
        "tables": {
            "A": 0,
            "B": 0,
            "C": 0,
            "total": 0
        },
        "total_cost": 0
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, "UNKNOWN"),
    "objective_value": objective_value,
    "example_output": output
}, separators=(",", ":")))