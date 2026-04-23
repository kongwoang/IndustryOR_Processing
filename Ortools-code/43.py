import json
from ortools.linear_solver import pywraplp

inp = {
    "requirements": {
        "raw_material_A_min_pieces": 240,
        "raw_material_B_min_kg": 80,
        "raw_material_C_min_tons": 120
    },
    "warehouses": {
        "A": {
            "raw_material_A_per_truck": 4,
            "raw_material_B_per_truck": 2,
            "raw_material_C_per_truck": 6,
            "cost_per_truck": 200
        },
        "B": {
            "raw_material_A_per_truck": 7,
            "raw_material_B_per_truck": 2,
            "raw_material_C_per_truck": 2,
            "cost_per_truck": 160
        }
    }
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    solver = pywraplp.Solver.CreateSolver("CBC")

wa = inp["warehouses"]["A"]
wb = inp["warehouses"]["B"]
req = inp["requirements"]

x_a = solver.IntVar(0.0, solver.infinity(), "x_a")
x_b = solver.IntVar(0.0, solver.infinity(), "x_b")

solver.Add(
    wa["raw_material_A_per_truck"] * x_a + wb["raw_material_A_per_truck"] * x_b
    >= req["raw_material_A_min_pieces"]
)
solver.Add(
    wa["raw_material_B_per_truck"] * x_a + wb["raw_material_B_per_truck"] * x_b
    >= req["raw_material_B_min_kg"]
)
solver.Add(
    wa["raw_material_C_per_truck"] * x_a + wb["raw_material_C_per_truck"] * x_b
    >= req["raw_material_C_min_tons"]
)

solver.Minimize(
    wa["cost_per_truck"] * x_a + wb["cost_per_truck"] * x_b
)

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
    output = {
        "trucks_from_warehouse_A": int(round(x_a.solution_value())),
        "trucks_from_warehouse_B": int(round(x_b.solution_value()))
    }
    objective_value = solver.Objective().Value()
    if abs(objective_value - round(objective_value)) < 1e-9:
        objective_value = int(round(objective_value))
else:
    output = {
        "trucks_from_warehouse_A": None,
        "trucks_from_warehouse_B": None
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))