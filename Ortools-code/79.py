import json
from ortools.linear_solver import pywraplp

inp = {
    "products": ["tables", "chairs", "bookshelves"],
    "selling_price": {
        "tables": 200,
        "chairs": 50,
        "bookshelves": 150
    },
    "manufacturing_cost": {
        "tables": 120,
        "chairs": 20,
        "bookshelves": 90
    },
    "space": {
        "tables": 5,
        "chairs": 2,
        "bookshelves": 3
    },
    "max_space": 500,
    "min_production": {
        "tables": 10,
        "bookshelves": 20
    },
    "max_total_items": 200
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

products = inp["products"]
min_production = inp["min_production"]

x = {}
for p in products:
    lb = min_production.get(p, 0)
    x[p] = solver.IntVar(lb, solver.infinity(), p)

solver.Add(sum(inp["space"][p] * x[p] for p in products) <= inp["max_space"])
solver.Add(sum(x[p] for p in products) <= inp["max_total_items"])

profit = {
    p: inp["selling_price"][p] - inp["manufacturing_cost"][p]
    for p in products
}
solver.Maximize(sum(profit[p] * x[p] for p in products))

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
        "tables": int(round(x["tables"].solution_value())),
        "chairs": int(round(x["chairs"].solution_value())),
        "bookshelves": int(round(x["bookshelves"].solution_value()))
    }
    objective_value = int(round(solver.Objective().Value()))
else:
    output = {
        "tables": 0,
        "chairs": 0,
        "bookshelves": 0
    }
    objective_value = None

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result))