import json
from ortools.linear_solver import pywraplp

inp = {
    "candidate_areas": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
    "coverage": {
        "A": ["A", "C", "E", "G", "H", "I"],
        "B": ["B", "H", "I"],
        "C": ["A", "C", "G", "H", "I"],
        "D": ["D", "J"],
        "E": ["A", "E", "G"],
        "F": ["F", "J", "K"],
        "G": ["A", "C", "E", "G"],
        "H": ["A", "B", "C", "H", "I"],
        "I": ["A", "B", "C", "H", "I"],
        "J": ["D", "F", "J", "K", "L"],
        "K": ["F", "J", "K", "L"],
        "L": ["J", "K", "L"]
    }
}

areas = inp["candidate_areas"]
coverage = inp["coverage"]

# First stage: minimize the number of stores
solver = pywraplp.Solver.CreateSolver("SCIP")
x = {a: solver.BoolVar(f"x_{a}") for a in areas}

for demand_area in areas:
    covering_sites = [site for site in areas if demand_area in coverage[site]]
    solver.Add(sum(x[site] for site in covering_sites) >= 1)

solver.Minimize(sum(x[a] for a in areas))
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    min_store_count = int(round(solver.Objective().Value()))

    # Second stage: among all minimum-cardinality solutions,
    # choose one with the smallest sum of alphabetical indices
    solver2 = pywraplp.Solver.CreateSolver("SCIP")
    y = {a: solver2.BoolVar(f"y_{a}") for a in areas}

    for demand_area in areas:
        covering_sites = [site for site in areas if demand_area in coverage[site]]
        solver2.Add(sum(y[site] for site in covering_sites) >= 1)

    solver2.Add(sum(y[a] for a in areas) == min_store_count)

    weights = {a: i + 1 for i, a in enumerate(areas)}
    solver2.Minimize(sum(weights[a] * y[a] for a in areas))
    status2 = solver2.Solve()

    if status2 == pywraplp.Solver.OPTIMAL:
        selected_areas = [a for a in areas if y[a].solution_value() > 0.5]
        output = {
            "minimum_number_of_stores": min_store_count,
            "store_locations": selected_areas
        }
        result = {
            "status": "OPTIMAL",
            "objective_value": min_store_count,
            "example_output": output
        }
    else:
        output = {
            "minimum_number_of_stores": None,
            "store_locations": []
        }
        result = {
            "status": "SECOND_STAGE_FAILED",
            "objective_value": min_store_count,
            "example_output": output
        }
else:
    output = {
        "minimum_number_of_stores": None,
        "store_locations": []
    }
    result = {
        "status": "INFEASIBLE_OR_NOT_SOLVED",
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))