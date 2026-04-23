import json
from ortools.linear_solver import pywraplp

inp = {
    "inventory": {
        "shirts": 200,
        "pants": 100
    },
    "packages": {
        "A": {
            "shirts": 1,
            "pants": 2,
            "price": 30,
            "minimum_to_sell": 20
        },
        "B": {
            "shirts": 3,
            "pants": 1,
            "price": 50,
            "minimum_to_sell": 10
        }
    },
    "objective": "maximize_revenue"
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

A = solver.IntVar(inp["packages"]["A"]["minimum_to_sell"], solver.infinity(), "A")
B = solver.IntVar(inp["packages"]["B"]["minimum_to_sell"], solver.infinity(), "B")

solver.Add(
    inp["packages"]["A"]["shirts"] * A + inp["packages"]["B"]["shirts"] * B
    <= inp["inventory"]["shirts"]
)
solver.Add(
    inp["packages"]["A"]["pants"] * A + inp["packages"]["B"]["pants"] * B
    <= inp["inventory"]["pants"]
)

solver.Maximize(
    inp["packages"]["A"]["price"] * A + inp["packages"]["B"]["price"] * B
)

status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    output = {
        "package_A_to_sell": int(round(A.solution_value())),
        "package_B_to_sell": int(round(B.solution_value()))
    }
    result = {
        "status": "OPTIMAL",
        "objective_value": solver.Objective().Value(),
        "example_output": output
    }
elif status == pywraplp.Solver.FEASIBLE:
    output = {
        "package_A_to_sell": int(round(A.solution_value())),
        "package_B_to_sell": int(round(B.solution_value()))
    }
    result = {
        "status": "FEASIBLE",
        "objective_value": solver.Objective().Value(),
        "example_output": output
    }
else:
    output = {
        "package_A_to_sell": None,
        "package_B_to_sell": None
    }
    status_map = {
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
    }
    result = {
        "status": status_map.get(status, "UNKNOWN"),
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))