# Source: :contentReference[oaicite:0]{index=0}
import json
from ortools.linear_solver import pywraplp

inp = {
    "products": {
        "Meaties": {
            "price_per_pack": 2.8,
            "grains_lbs_per_pack": 2.0,
            "meat_lbs_per_pack": 3.0,
            "variable_cost_per_pack": 0.25
        },
        "Yummies": {
            "price_per_pack": 2.0,
            "grains_lbs_per_pack": 3.0,
            "meat_lbs_per_pack": 1.5,
            "variable_cost_per_pack": 0.2
        }
    },
    "resources": {
        "grains": {
            "max_lbs": 400000,
            "cost_per_lb": 0.2
        },
        "meat": {
            "max_lbs": 300000,
            "cost_per_lb": 0.5
        },
        "meaties_capacity_packs": 90000
    }
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Failed to create GLOP solver.")

x = solver.NumVar(0.0, solver.infinity(), "Meaties")
y = solver.NumVar(0.0, solver.infinity(), "Yummies")

m = inp["products"]["Meaties"]
u = inp["products"]["Yummies"]
r = inp["resources"]

solver.Add(
    m["grains_lbs_per_pack"] * x + u["grains_lbs_per_pack"] * y <= r["grains"]["max_lbs"]
)
solver.Add(
    m["meat_lbs_per_pack"] * x + u["meat_lbs_per_pack"] * y <= r["meat"]["max_lbs"]
)
solver.Add(x <= r["meaties_capacity_packs"])

revenue = m["price_per_pack"] * x + u["price_per_pack"] * y
raw_material_cost = (
    r["grains"]["cost_per_lb"] * (m["grains_lbs_per_pack"] * x + u["grains_lbs_per_pack"] * y)
    + r["meat"]["cost_per_lb"] * (m["meat_lbs_per_pack"] * x + u["meat_lbs_per_pack"] * y)
)
mixing_packaging_cost = m["variable_cost_per_pack"] * x + u["variable_cost_per_pack"] * y
profit = revenue - raw_material_cost - mixing_packaging_cost

solver.Maximize(profit)
status_code = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

def clean_num(v):
    if v is None:
        return None
    if abs(v - round(v)) <= 1e-6:
        return int(round(v))
    return round(v, 6)

if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    meaties = x.solution_value()
    yummies = y.solution_value()
    grains_used = m["grains_lbs_per_pack"] * meaties + u["grains_lbs_per_pack"] * yummies
    meat_used = m["meat_lbs_per_pack"] * meaties + u["meat_lbs_per_pack"] * yummies
    revenue_val = m["price_per_pack"] * meaties + u["price_per_pack"] * yummies
    raw_material_cost_val = (
        r["grains"]["cost_per_lb"] * grains_used + r["meat"]["cost_per_lb"] * meat_used
    )
    mixing_packaging_cost_val = (
        m["variable_cost_per_pack"] * meaties + u["variable_cost_per_pack"] * yummies
    )
    profit_val = revenue_val - raw_material_cost_val - mixing_packaging_cost_val

    output = {
        "production_packs": {
            "Meaties": clean_num(meaties),
            "Yummies": clean_num(yummies)
        },
        "resource_usage": {
            "grains_lbs": clean_num(grains_used),
            "meat_lbs": clean_num(meat_used)
        },
        "financials": {
            "revenue": clean_num(revenue_val),
            "raw_material_cost": clean_num(raw_material_cost_val),
            "mixing_packaging_cost": clean_num(mixing_packaging_cost_val),
            "profit": clean_num(profit_val)
        }
    }
    objective_value = clean_num(solver.Objective().Value())
else:
    output = {
        "production_packs": {
            "Meaties": None,
            "Yummies": None
        },
        "resource_usage": {
            "grains_lbs": None,
            "meat_lbs": None
        },
        "financials": {
            "revenue": None,
            "raw_material_cost": None,
            "mixing_packaging_cost": None,
            "profit": None
        }
    }
    objective_value = None

print(json.dumps({
    "status": status_map.get(status_code, str(status_code)),
    "objective_value": objective_value,
    "example_output": output
}, separators=(",", ":")))