import json
from ortools.linear_solver import pywraplp

inp = {
    "parts": 10,
    "machines": ["A", "B", "C"],
    "processing_costs": {
        "A": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "B": [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
        "C": [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    },
    "setup_costs": {"A": 100, "B": 135, "C": 200},
    "fixed_assignments": {"3": "A", "4": "B", "5": "C"},
    "logical_rules": [
        "part 2 is assigned to A if and only if part 1 is not assigned to A"
    ],
    "max_parts_on_machine": {"C": 3}
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

parts = list(range(1, inp["parts"] + 1))
machines = inp["machines"]
costs = inp["processing_costs"]
setup_costs = inp["setup_costs"]

x = {}
for i in parts:
    for m in machines:
        x[i, m] = solver.BoolVar(f"x_{i}_{m}")

y = {m: solver.BoolVar(f"y_{m}") for m in machines}

# Each part is assigned to exactly one machine
for i in parts:
    solver.Add(sum(x[i, m] for m in machines) == 1)

# Activate setup variable if any part is assigned to a machine
for m in machines:
    for i in parts:
        solver.Add(x[i, m] <= y[m])

# Fixed assignments: part 3->A, part 4->B, part 5->C
for part_str, req_machine in inp["fixed_assignments"].items():
    i = int(part_str)
    solver.Add(x[i, req_machine] == 1)

# Logical rule:
# If part 1 is on A, then part 2 is on B or C;
# if part 1 is on B or C, then part 2 is on A.
# Equivalent to x[1,A] + x[2,A] = 1.
solver.Add(x[1, "A"] + x[2, "A"] == 1)

# Number of part types processed on C does not exceed 3
solver.Add(sum(x[i, "C"] for i in parts) <= inp["max_parts_on_machine"]["C"])

# Objective: minimize processing cost + setup cost
processing_expr = sum(costs[m][i - 1] * x[i, m] for i in parts for m in machines)
setup_expr = sum(setup_costs[m] * y[m] for m in machines)
solver.Minimize(processing_expr + setup_expr)

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
    assignments = {}
    machines_used = []
    total_processing_cost = 0

    for i in parts:
        for m in machines:
            if x[i, m].solution_value() > 0.5:
                assignments[str(i)] = m
                total_processing_cost += costs[m][i - 1]
                break

    for m in machines:
        if y[m].solution_value() > 0.5:
            machines_used.append(m)

    total_setup_cost = sum(setup_costs[m] for m in machines_used)
    total_cost = total_processing_cost + total_setup_cost

    output = {
        "assignments": assignments,
        "machines_used": machines_used,
        "total_processing_cost": total_processing_cost,
        "total_setup_cost": total_setup_cost,
        "total_cost": total_cost
    }

    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": solver.Objective().Value(),
        "example_output": output
    }
else:
    output = {
        "assignments": {},
        "machines_used": [],
        "total_processing_cost": None,
        "total_setup_cost": None,
        "total_cost": None
    }
    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))