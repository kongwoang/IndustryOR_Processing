import json
from ortools.linear_solver import pywraplp

inp = {
    "months": ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "products": {
        "I": {
            "demand": {
                "Jul": 30000,
                "Aug": 30000,
                "Sep": 30000,
                "Oct": 100000,
                "Nov": 100000,
                "Dec": 100000
            },
            "production_cost": {
                "Jul": 4.5,
                "Aug": 4.5,
                "Sep": 4.5,
                "Oct": 4.5,
                "Nov": 4.5,
                "Dec": 4.5
            },
            "volume_per_unit_m3": 0.2
        },
        "II": {
            "demand": {
                "Jul": 15000,
                "Aug": 15000,
                "Sep": 15000,
                "Oct": 50000,
                "Nov": 50000,
                "Dec": 50000
            },
            "production_cost": {
                "Jul": 7.0,
                "Aug": 7.0,
                "Sep": 7.0,
                "Oct": 7.0,
                "Nov": 7.0,
                "Dec": 7.0
            },
            "volume_per_unit_m3": 0.4
        }
    },
    "monthly_combined_capacity_units": 120000,
    "warehouse": {
        "internal_capacity_m3": 15000,
        "internal_cost_per_m3_per_month": 1.0,
        "external_cost_per_m3_per_month": 1.5
    },
    "initial_inventory_units": {
        "I": 0,
        "II": 0
    }
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("GLOP solver is not available.")

months = inp["months"]
products = list(inp["products"].keys())

x = {(p, m): solver.NumVar(0.0, solver.infinity(), f"x_{p}_{m}") for p in products for m in months}
inv = {(p, m): solver.NumVar(0.0, solver.infinity(), f"inv_{p}_{m}") for p in products for m in months}
w_in = {m: solver.NumVar(0.0, inp["warehouse"]["internal_capacity_m3"], f"w_in_{m}") for m in months}
w_out = {m: solver.NumVar(0.0, solver.infinity(), f"w_out_{m}") for m in months}

for m in months:
    solver.Add(sum(x[p, m] for p in products) <= inp["monthly_combined_capacity_units"])

prev_inv = {p: inp["initial_inventory_units"][p] for p in products}
for m in months:
    for p in products:
        solver.Add(
            inv[p, m] == prev_inv[p] + x[p, m] - inp["products"][p]["demand"][m]
        )
    total_volume = sum(inp["products"][p]["volume_per_unit_m3"] * inv[p, m] for p in products)
    solver.Add(w_in[m] + w_out[m] == total_volume)
    prev_inv = {p: inv[p, m] for p in products}

objective = solver.Objective()
for m in months:
    for p in products:
        objective.SetCoefficient(x[p, m], inp["products"][p]["production_cost"][m])
    objective.SetCoefficient(w_in[m], inp["warehouse"]["internal_cost_per_m3_per_month"])
    objective.SetCoefficient(w_out[m], inp["warehouse"]["external_cost_per_m3_per_month"])
objective.SetMinimization()

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

def clean_num(v, ndigits=6):
    v = round(float(v), ndigits)
    if abs(v - round(v)) < 1e-6:
        return int(round(v))
    return v

output = {
    "production_units": {},
    "ending_inventory_units": {},
    "warehouse_usage_m3": {},
    "total_production_cost": None,
    "total_inventory_cost": None,
    "total_cost": None
}

if status in {"OPTIMAL", "FEASIBLE"}:
    production_units = {}
    ending_inventory_units = {}
    warehouse_usage_m3 = {}

    total_production_cost = 0.0
    total_inventory_cost = 0.0

    for m in months:
        production_units[m] = {}
        ending_inventory_units[m] = {}
        for p in products:
            xv = x[p, m].solution_value()
            iv = inv[p, m].solution_value()
            production_units[m][p] = clean_num(xv)
            ending_inventory_units[m][p] = clean_num(iv)
            total_production_cost += inp["products"][p]["production_cost"][m] * xv

        win_v = w_in[m].solution_value()
        wout_v = w_out[m].solution_value()
        warehouse_usage_m3[m] = {
            "internal": clean_num(win_v),
            "external": clean_num(wout_v)
        }
        total_inventory_cost += (
            inp["warehouse"]["internal_cost_per_m3_per_month"] * win_v
            + inp["warehouse"]["external_cost_per_m3_per_month"] * wout_v
        )

    output = {
        "production_units": production_units,
        "ending_inventory_units": ending_inventory_units,
        "warehouse_usage_m3": warehouse_usage_m3,
        "total_production_cost": clean_num(total_production_cost, 2),
        "total_inventory_cost": clean_num(total_inventory_cost, 2),
        "total_cost": clean_num(total_production_cost + total_inventory_cost, 2)
    }

result = {
    "status": status,
    "objective_value": clean_num(solver.Objective().Value(), 2) if status in {"OPTIMAL", "FEASIBLE"} else None,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))