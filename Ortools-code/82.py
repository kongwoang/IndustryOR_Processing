import json
from ortools.linear_solver import pywraplp

inp = {
    "total_area_m2": 5000,
    "rent_ratio": 0.2,
    "store_types": [
        {
            "code": 1,
            "store_type": "Jewelry",
            "area_per_shop_m2": 250,
            "min_shops": 1,
            "max_shops": 3,
            "annual_profit_per_shop_10k_yuan": {
                "1": 9,
                "2": 8,
                "3": 7
            }
        },
        {
            "code": 2,
            "store_type": "Shoes & Hats",
            "area_per_shop_m2": 350,
            "min_shops": 1,
            "max_shops": 2,
            "annual_profit_per_shop_10k_yuan": {
                "1": 10,
                "2": 9
            }
        },
        {
            "code": 3,
            "store_type": "General Merchandise",
            "area_per_shop_m2": 800,
            "min_shops": 1,
            "max_shops": 3,
            "annual_profit_per_shop_10k_yuan": {
                "1": 27,
                "2": 21,
                "3": 20
            }
        },
        {
            "code": 4,
            "store_type": "Bookstore",
            "area_per_shop_m2": 400,
            "min_shops": 0,
            "max_shops": 2,
            "annual_profit_per_shop_10k_yuan": {
                "1": 16,
                "2": 10
            }
        },
        {
            "code": 5,
            "store_type": "Catering",
            "area_per_shop_m2": 500,
            "min_shops": 1,
            "max_shops": 3,
            "annual_profit_per_shop_10k_yuan": {
                "1": 17,
                "2": 15,
                "3": 12
            }
        }
    ]
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("Failed to create solver.")

store_types = inp["store_types"]
y = {}

for i, s in enumerate(store_types):
    allowed_counts = range(s["min_shops"], s["max_shops"] + 1)
    for n in allowed_counts:
        y[(i, n)] = solver.BoolVar(f"y_{i}_{n}")
    solver.Add(sum(y[(i, n)] for n in allowed_counts) == 1)

solver.Add(
    sum(
        s["area_per_shop_m2"] * n * y[(i, n)]
        for i, s in enumerate(store_types)
        for n in range(s["min_shops"], s["max_shops"] + 1)
    ) <= inp["total_area_m2"]
)

total_profit_expr = sum(
    (0 if n == 0 else n * s["annual_profit_per_shop_10k_yuan"][str(n)]) * y[(i, n)]
    for i, s in enumerate(store_types)
    for n in range(s["min_shops"], s["max_shops"] + 1)
)

solver.Maximize(total_profit_expr)

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
    store_counts = []
    total_area_used = 0
    total_annual_profit = 0

    for i, s in enumerate(store_types):
        chosen_n = None
        for n in range(s["min_shops"], s["max_shops"] + 1):
            if y[(i, n)].solution_value() > 0.5:
                chosen_n = n
                break
        if chosen_n is None:
            chosen_n = s["min_shops"]

        store_counts.append({
            "code": s["code"],
            "store_type": s["store_type"],
            "shops": int(chosen_n)
        })

        total_area_used += s["area_per_shop_m2"] * chosen_n
        if chosen_n > 0:
            total_annual_profit += chosen_n * s["annual_profit_per_shop_10k_yuan"][str(chosen_n)]

    total_rent_income = inp["rent_ratio"] * total_annual_profit

    output = {
        "store_counts": store_counts,
        "total_area_used_m2": total_area_used,
        "total_annual_profit_10k_yuan": total_annual_profit,
        "total_rent_income_10k_yuan": total_rent_income
    }

    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": total_rent_income,
        "example_output": output
    }
else:
    output = {
        "store_counts": [],
        "total_area_used_m2": None,
        "total_annual_profit_10k_yuan": None,
        "total_rent_income_10k_yuan": None
    }
    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))