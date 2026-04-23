import json
from ortools.linear_solver import pywraplp

inp = {
    "container_types": [
        {
            "code": 1,
            "volume_cm3": 1500,
            "demand_units": 500,
            "variable_cost_per_unit": 5.0
        },
        {
            "code": 2,
            "volume_cm3": 2500,
            "demand_units": 550,
            "variable_cost_per_unit": 8.0
        },
        {
            "code": 3,
            "volume_cm3": 4000,
            "demand_units": 700,
            "variable_cost_per_unit": 10.0
        },
        {
            "code": 4,
            "volume_cm3": 6000,
            "demand_units": 900,
            "variable_cost_per_unit": 12.0
        },
        {
            "code": 5,
            "volume_cm3": 9000,
            "demand_units": 400,
            "variable_cost_per_unit": 16.0
        },
        {
            "code": 6,
            "volume_cm3": 12000,
            "demand_units": 300,
            "variable_cost_per_unit": 18.0
        }
    ],
    "fixed_setup_cost": 1200.0
}

container_types = inp["container_types"]
fixed_setup_cost = inp["fixed_setup_cost"]

codes = [ct["code"] for ct in container_types]
demand = {ct["code"]: int(ct["demand_units"]) for ct in container_types}
var_cost = {ct["code"]: float(ct["variable_cost_per_unit"]) for ct in container_types}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

# Decision variables
x = {i: solver.IntVar(0, solver.infinity(), f"x_{i}") for i in codes}  # production of type i
y = {i: solver.BoolVar(f"y_{i}") for i in codes}  # whether equipment i is activated

# a[i, j] = units of type i used to satisfy demand of type j, only allowed if i >= j
a = {}
for i in codes:
    for j in codes:
        if i >= j:
            a[(i, j)] = solver.IntVar(0, solver.infinity(), f"a_{i}_{j}")

# Constraints: every demand must be fully met
for j in codes:
    solver.Add(solver.Sum(a[(i, j)] for i in codes if i >= j) == demand[j])

# Constraints: allocation from each produced type cannot exceed production
for i in codes:
    solver.Add(solver.Sum(a[(i, j)] for j in codes if i >= j) == x[i])

# Activation linking constraints
for i in codes:
    max_useful_production = sum(demand[j] for j in codes if j <= i)
    solver.Add(x[i] <= max_useful_production * y[i])

# Objective: minimize variable production cost + fixed setup cost
objective = solver.Objective()
for i in codes:
    objective.SetCoefficient(x[i], var_cost[i])
    objective.SetCoefficient(y[i], fixed_setup_cost)
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

output = {
    "production_units": {str(i): 0 for i in codes},
    "equipment_activated": {str(i): 0 for i in codes},
    "allocation_units": {},
    "total_variable_cost": None,
    "total_fixed_cost": None,
    "total_cost": None,
}

for i in codes:
    output["allocation_units"][str(i)] = {}
    for j in codes:
        if i >= j:
            output["allocation_units"][str(i)][str(j)] = 0

objective_value = None

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    for i in codes:
        output["production_units"][str(i)] = int(round(x[i].solution_value()))
        output["equipment_activated"][str(i)] = int(round(y[i].solution_value()))
        for j in codes:
            if i >= j:
                output["allocation_units"][str(i)][str(j)] = int(round(a[(i, j)].solution_value()))

    total_variable_cost = sum(var_cost[i] * output["production_units"][str(i)] for i in codes)
    total_fixed_cost = sum(fixed_setup_cost * output["equipment_activated"][str(i)] for i in codes)
    total_cost = total_variable_cost + total_fixed_cost

    output["total_variable_cost"] = float(total_variable_cost)
    output["total_fixed_cost"] = float(total_fixed_cost)
    output["total_cost"] = float(total_cost)
    objective_value = float(solver.Objective().Value())

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))