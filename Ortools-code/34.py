import json
from ortools.linear_solver import pywraplp

inp = {
    "container": {
        "max_weight_tons": 60,
        "min_weight_tons": 18
    },
    "goods": {
        "A": {
            "quantity": 120,
            "weight_tons": 0.5
        },
        "B": {
            "quantity": 90,
            "weight_tons": 1.0
        },
        "C": {
            "quantity": 300,
            "weight_tons": 0.4
        },
        "D": {
            "quantity": 90,
            "weight_tons": 0.6
        },
        "E": {
            "quantity": 120,
            "weight_tons": 0.65
        }
    },
    "per_used_container_requirements": {
        "D_min_units": 12
    },
    "packaging_requirements": [
        {
            "base_good": "A",
            "paired_good": "C",
            "min_paired_units_per_base_unit": 1
        }
    ],
    "objective": "minimize_number_of_containers"
}

goods = list(inp["goods"].keys())
max_weight = inp["container"]["max_weight_tons"]
min_weight = inp["container"]["min_weight_tons"]
d_min = inp["per_used_container_requirements"]["D_min_units"]

total_units = sum(inp["goods"][g]["quantity"] for g in goods)
max_containers = total_units
if "D" in inp["goods"] and d_min > 0:
    max_containers = min(max_containers, inp["goods"]["D"]["quantity"] // d_min)
if max_containers <= 0:
    max_containers = total_units

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    solver = pywraplp.Solver.CreateSolver("CBC")
if solver is None:
    raise RuntimeError("No suitable MILP solver available in OR-Tools.")

containers = range(max_containers)

x = {}
for g in goods:
    qty = inp["goods"][g]["quantity"]
    for j in containers:
        x[g, j] = solver.IntVar(0, qty, f"x_{g}_{j}")

y = {j: solver.IntVar(0, 1, f"y_{j}") for j in containers}

for g in goods:
    solver.Add(sum(x[g, j] for j in containers) == inp["goods"][g]["quantity"])

for j in containers:
    total_weight_j = sum(inp["goods"][g]["weight_tons"] * x[g, j] for g in goods)
    solver.Add(total_weight_j <= max_weight * y[j])
    solver.Add(total_weight_j >= min_weight * y[j])

    for g in goods:
        solver.Add(x[g, j] <= inp["goods"][g]["quantity"] * y[j])

    if "D" in inp["goods"] and d_min > 0:
        solver.Add(x["D", j] >= d_min * y[j])

    for req in inp["packaging_requirements"]:
        base_good = req["base_good"]
        paired_good = req["paired_good"]
        ratio = req["min_paired_units_per_base_unit"]
        solver.Add(x[paired_good, j] >= ratio * x[base_good, j])

for j in range(max_containers - 1):
    solver.Add(y[j] >= y[j + 1])

solver.Minimize(sum(y[j] for j in containers))

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
    used = [j for j in containers if y[j].solution_value() > 0.5]
    output = {
        "container_count": len(used),
        "containers": []
    }
    for idx, j in enumerate(used, start=1):
        loads = {g: int(round(x[g, j].solution_value())) for g in goods}
        total_weight_j = sum(inp["goods"][g]["weight_tons"] * loads[g] for g in goods)
        output["containers"].append({
            "container_id": idx,
            "loads": loads,
            "total_weight_tons": total_weight_j
        })
    objective_value = int(round(solver.Objective().Value()))
else:
    output = {
        "container_count": 0,
        "containers": []
    }
    objective_value = None

result = {
    "status": status_str,
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))