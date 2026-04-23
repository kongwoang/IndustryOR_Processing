# Source problem statement: :contentReference[oaicite:0]{index=0}
import json
from ortools.linear_solver import pywraplp

inp = {
    "commodities": ["steel", "engines", "electronic_components", "plastic"],
    "world_prices_klunz": {
        "steel": 500,
        "engines": 1500,
        "electronic_components": 300,
        "plastic": 1200
    },
    "production_requirements": {
        "steel": {
            "inputs": {
                "engines": 0.02,
                "plastic": 0.01
            },
            "imported_goods_klunz": 250,
            "labor_person_months": 6
        },
        "engines": {
            "inputs": {
                "steel": 0.8,
                "electronic_components": 0.15,
                "plastic": 0.11
            },
            "imported_goods_klunz": 300,
            "labor_person_months": 12
        },
        "electronic_components": {
            "inputs": {
                "steel": 0.01,
                "engines": 0.01,
                "plastic": 0.05
            },
            "imported_goods_klunz": 50,
            "labor_person_months": 6
        },
        "plastic": {
            "inputs": {
                "engines": 0.03,
                "steel": 0.2,
                "electronic_components": 0.05
            },
            "imported_goods_klunz": 300,
            "labor_person_months": 24
        }
    },
    "capacity_upper_bounds": {
        "engines": 650000,
        "plastic": 60000
    },
    "total_labor_person_months": 830000
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("OR-Tools GLOP solver is unavailable.")

commodities = inp["commodities"]
prices = inp["world_prices_klunz"]
reqs = inp["production_requirements"]
caps = inp["capacity_upper_bounds"]

production = {c: solver.NumVar(0.0, solver.infinity(), f"prod_{c}") for c in commodities}
net_exports = {c: solver.NumVar(0.0, solver.infinity(), f"net_export_{c}") for c in commodities}

for c in commodities:
    solver.Add(
        net_exports[c] ==
        production[c] - sum(reqs[j]["inputs"].get(c, 0.0) * production[j] for j in commodities)
    )

for c, ub in caps.items():
    solver.Add(production[c] <= ub)

solver.Add(
    sum(reqs[c]["labor_person_months"] * production[c] for c in commodities)
    <= inp["total_labor_person_months"]
)

objective = solver.Objective()
for c in commodities:
    objective.SetCoefficient(net_exports[c], prices[c])
    objective.SetCoefficient(production[c], -reqs[c]["imported_goods_klunz"])
objective.SetMaximization()

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

def r(x):
    return round(float(x), 6)

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    imported_goods_cost = sum(
        reqs[c]["imported_goods_klunz"] * production[c].solution_value()
        for c in commodities
    )
    labor_used = sum(
        reqs[c]["labor_person_months"] * production[c].solution_value()
        for c in commodities
    )
    gdp = objective.Value()

    output = {
        "production_units": {c: r(production[c].solution_value()) for c in commodities},
        "net_exports_units": {c: r(net_exports[c].solution_value()) for c in commodities},
        "imported_goods_cost_klunz": r(imported_goods_cost),
        "labor_used_person_months": r(labor_used),
        "gdp_klunz": r(gdp)
    }
    objective_value = r(gdp)
else:
    output = {
        "production_units": {c: 0.0 for c in commodities},
        "net_exports_units": {c: 0.0 for c in commodities},
        "imported_goods_cost_klunz": 0.0,
        "labor_used_person_months": 0.0,
        "gdp_klunz": 0.0
    }
    objective_value = 0.0

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))