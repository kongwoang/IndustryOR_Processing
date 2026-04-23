import json
from ortools.linear_solver import pywraplp

inp = {
    "products": ["I", "II", "III"],
    "machines": {
        "A": {
            "A1": {
                "available_time": 6000,
                "hourly_cost": 0.05,
                "processing_time": {
                    "I": 5,
                    "II": 10
                }
            },
            "A2": {
                "available_time": 10000,
                "hourly_cost": 0.03,
                "processing_time": {
                    "I": 7,
                    "II": 9,
                    "III": 12
                }
            }
        },
        "B": {
            "B1": {
                "available_time": 4000,
                "hourly_cost": 0.06,
                "processing_time": {
                    "I": 6,
                    "II": 8
                }
            },
            "B2": {
                "available_time": 7000,
                "hourly_cost": 0.11,
                "processing_time": {
                    "I": 4,
                    "III": 11
                }
            },
            "B3": {
                "available_time": 4000,
                "hourly_cost": 0.05,
                "processing_time": {
                    "I": 7
                }
            }
        }
    },
    "raw_material_cost": {
        "I": 0.25,
        "II": 0.35,
        "III": 0.5
    },
    "unit_price": {
        "I": 1.25,
        "II": 2.0,
        "III": 2.8
    }
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    solver = pywraplp.Solver.CreateSolver("CBC")
if solver is None:
    raise RuntimeError("Failed to create an integer solver (SCIP/CBC).")

products = inp["products"]
machines_a = inp["machines"]["A"]
machines_b = inp["machines"]["B"]

routes = {
    "I": [("A1", "B1"), ("A1", "B2"), ("A1", "B3"), ("A2", "B1"), ("A2", "B2"), ("A2", "B3")],
    "II": [("A1", "B1"), ("A2", "B1")],
    "III": [("A2", "B2")]
}

x = {}
for p in products:
    for a, b in routes[p]:
        x[(p, a, b)] = solver.IntVar(0, solver.infinity(), f"x_{p}_{a}_{b}")

# Capacity constraints for stage A
for a_name, a_data in machines_a.items():
    solver.Add(
        solver.Sum(
            a_data["processing_time"][p] * x[(p, a_name, b)]
            for p in products
            for (a, b) in routes[p]
            if a == a_name
        ) <= a_data["available_time"]
    )

# Capacity constraints for stage B
for b_name, b_data in machines_b.items():
    solver.Add(
        solver.Sum(
            b_data["processing_time"][p] * x[(p, a, b_name)]
            for p in products
            for (a, b) in routes[p]
            if b == b_name
        ) <= b_data["available_time"]
    )

# Objective: maximize profit
objective = solver.Objective()
for p in products:
    for a, b in routes[p]:
        revenue = inp["unit_price"][p]
        raw_cost = inp["raw_material_cost"][p]
        stage_a_cost = machines_a[a]["processing_time"][p] * machines_a[a]["hourly_cost"]
        stage_b_cost = machines_b[b]["processing_time"][p] * machines_b[b]["hourly_cost"]
        unit_profit = revenue - raw_cost - stage_a_cost - stage_b_cost
        objective.SetCoefficient(x[(p, a, b)], unit_profit)
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
status = status_map.get(status_code, str(status_code))

def r(v, ndigits=6):
    return round(float(v), ndigits)

output = {
    "production_quantity": {
        "I": 0,
        "II": 0,
        "III": 0
    },
    "route_allocation": {
        "I": {
            "A1-B1": 0,
            "A1-B2": 0,
            "A1-B3": 0,
            "A2-B1": 0,
            "A2-B2": 0,
            "A2-B3": 0
        },
        "II": {
            "A1-B1": 0,
            "A2-B1": 0
        },
        "III": {
            "A2-B2": 0
        }
    },
    "machine_usage": {
        "A1": 0,
        "A2": 0,
        "B1": 0,
        "B2": 0,
        "B3": 0
    }
}

if status in {"OPTIMAL", "FEASIBLE"}:
    for p in products:
        total_p = 0.0
        for a, b in routes[p]:
            val = x[(p, a, b)].solution_value()
            output["route_allocation"][p][f"{a}-{b}"] = int(round(val))
            total_p += val
        output["production_quantity"][p] = int(round(total_p))

    for a_name, a_data in machines_a.items():
        usage = 0.0
        for p in products:
            for a, b in routes[p]:
                if a == a_name:
                    usage += a_data["processing_time"][p] * x[(p, a_name, b)].solution_value()
        output["machine_usage"][a_name] = r(usage)

    for b_name, b_data in machines_b.items():
        usage = 0.0
        for p in products:
            for a, b in routes[p]:
                if b == b_name:
                    usage += b_data["processing_time"][p] * x[(p, a, b_name)].solution_value()
        output["machine_usage"][b_name] = r(usage)

    objective_value = r(objective.Value(), 2)
else:
    objective_value = None

print(json.dumps({
    "status": status,
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))