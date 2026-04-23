import json
from ortools.linear_solver import pywraplp

inp = {
    "products": [
        {
            "name": "I",
            "raw_material_cost": 0.25,
            "sale_price": 1.25
        },
        {
            "name": "II",
            "raw_material_cost": 0.35,
            "sale_price": 2.0
        },
        {
            "name": "III",
            "raw_material_cost": 0.5,
            "sale_price": 2.8
        }
    ],
    "machines": [
        {
            "name": "A1",
            "stage": "A",
            "effective_hours": 6000,
            "full_capacity_operating_cost": 300
        },
        {
            "name": "A2",
            "stage": "A",
            "effective_hours": 10000,
            "full_capacity_operating_cost": 321
        },
        {
            "name": "B1",
            "stage": "B",
            "effective_hours": 4000,
            "full_capacity_operating_cost": 250
        },
        {
            "name": "B2",
            "stage": "B",
            "effective_hours": 7000,
            "full_capacity_operating_cost": 783
        },
        {
            "name": "B3",
            "stage": "B",
            "effective_hours": 4000,
            "full_capacity_operating_cost": 200
        }
    ],
    "processing_time": {
        "A1": {
            "I": 5,
            "II": 10
        },
        "A2": {
            "I": 7,
            "II": 9,
            "III": 12
        },
        "B1": {
            "I": 6,
            "II": 8
        },
        "B2": {
            "I": 4,
            "III": 11
        },
        "B3": {
            "I": 7
        }
    }
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("Failed to create SCIP solver.")

products = [p["name"] for p in inp["products"]]
product_data = {p["name"]: p for p in inp["products"]}
machines = {m["name"]: m for m in inp["machines"]}
processing_time = inp["processing_time"]

A_machines = [m["name"] for m in inp["machines"] if m["stage"] == "A"]
B_machines = [m["name"] for m in inp["machines"] if m["stage"] == "B"]

hour_cost = {
    m["name"]: m["full_capacity_operating_cost"] / m["effective_hours"]
    for m in inp["machines"]
}

xA = {}
xB = {}
q = {}

for p in products:
    q[p] = solver.IntVar(0.0, solver.infinity(), f"q_{p}")

for m in A_machines:
    for p in processing_time.get(m, {}):
        xA[(m, p)] = solver.IntVar(0.0, solver.infinity(), f"xA_{m}_{p}")

for m in B_machines:
    for p in processing_time.get(m, {}):
        xB[(m, p)] = solver.IntVar(0.0, solver.infinity(), f"xB_{m}_{p}")

# Flow balance: each finished unit must pass through both stage A and stage B.
for p in products:
    solver.Add(
        solver.Sum(xA[(m, p)] for m in A_machines if (m, p) in xA) == q[p]
    )
    solver.Add(
        solver.Sum(xB[(m, p)] for m in B_machines if (m, p) in xB) == q[p]
    )

# Machine capacity constraints.
for m in A_machines:
    solver.Add(
        solver.Sum(processing_time[m][p] * xA[(m, p)] for p in processing_time.get(m, {}))
        <= machines[m]["effective_hours"]
    )

for m in B_machines:
    solver.Add(
        solver.Sum(processing_time[m][p] * xB[(m, p)] for p in processing_time.get(m, {}))
        <= machines[m]["effective_hours"]
    )

# Objective: maximize profit = revenue - raw material costs - proportional machine operating costs.
revenue_minus_material = solver.Sum(
    (product_data[p]["sale_price"] - product_data[p]["raw_material_cost"]) * q[p]
    for p in products
)

machine_operating_cost = (
    solver.Sum(
        hour_cost[m] * processing_time[m][p] * xA[(m, p)]
        for (m, p) in xA
    )
    +
    solver.Sum(
        hour_cost[m] * processing_time[m][p] * xB[(m, p)]
        for (m, p) in xB
    )
)

solver.Maximize(revenue_minus_material - machine_operating_cost)

status_code = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}
status = status_map.get(status_code, str(status_code))

def round_float(x, nd=6):
    return round(float(x), nd)

output = {
    "total_profit": None,
    "product_quantities": {p: 0 for p in products},
    "stage_A_allocation": {
        m: {p: 0 for p in processing_time.get(m, {})}
        for m in A_machines
    },
    "stage_B_allocation": {
        m: {p: 0 for p in processing_time.get(m, {})}
        for m in B_machines
    }
}

objective_value = None

if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    objective_value = round_float(solver.Objective().Value())
    output["total_profit"] = objective_value
    output["product_quantities"] = {
        p: int(round(q[p].solution_value()))
        for p in products
    }
    output["stage_A_allocation"] = {
        m: {
            p: int(round(xA[(m, p)].solution_value()))
            for p in processing_time.get(m, {})
        }
        for m in A_machines
    }
    output["stage_B_allocation"] = {
        m: {
            p: int(round(xB[(m, p)].solution_value()))
            for p in processing_time.get(m, {})
        }
        for m in B_machines
    }

result = {
    "status": status,
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))