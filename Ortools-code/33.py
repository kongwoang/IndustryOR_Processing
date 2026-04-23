import json
from ortools.linear_solver import pywraplp

inp = {
    "items": [
        {
            "id": "painting_caillebotte",
            "description": "Painting by Caillebotte",
            "value": 25000
        },
        {
            "id": "bust_diocletian",
            "description": "Bust of Diocletian",
            "value": 5000
        },
        {
            "id": "vase_yuan_dynasty",
            "description": "Yuan dynasty Chinese vase",
            "value": 20000
        },
        {
            "id": "porsche_911",
            "description": "911 Porsche",
            "value": 40000
        },
        {
            "id": "diamond_1",
            "description": "Diamond 1",
            "value": 12000
        },
        {
            "id": "diamond_2",
            "description": "Diamond 2",
            "value": 12000
        },
        {
            "id": "diamond_3",
            "description": "Diamond 3",
            "value": 12000
        },
        {
            "id": "sofa_louis_xv",
            "description": "Louis XV sofa",
            "value": 3000
        },
        {
            "id": "jack_russell_dogs_pair",
            "description": "Two Jack Russell racing dogs (must stay together)",
            "value": 6000
        },
        {
            "id": "sculpture_200_ad",
            "description": "Sculpture from 200 AD",
            "value": 10000
        },
        {
            "id": "sailing_boat",
            "description": "Sailing boat",
            "value": 15000
        },
        {
            "id": "harley_davidson",
            "description": "Harley Davidson motorcycle",
            "value": 10000
        },
        {
            "id": "cavour_furniture",
            "description": "Piece of furniture once belonging to Cavour",
            "value": 13000
        }
    ]
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

items = inp["items"]
n = len(items)
values = [item["value"] for item in items]
total_value = sum(values)

x = [solver.BoolVar(f"x_{i}") for i in range(n)]
d = solver.IntVar(0, total_value, "absolute_difference")

son_1_total_expr = solver.Sum(values[i] * x[i] for i in range(n))
son_2_total_expr = total_value - son_1_total_expr

solver.Add(d >= son_1_total_expr - son_2_total_expr)
solver.Add(d >= son_2_total_expr - son_1_total_expr)

solver.Minimize(d)

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
    son_1_items = [items[i]["id"] for i in range(n) if x[i].solution_value() > 0.5]
    son_2_items = [items[i]["id"] for i in range(n) if x[i].solution_value() <= 0.5]
    son_1_total_value = int(round(sum(values[i] for i in range(n) if x[i].solution_value() > 0.5)))
    son_2_total_value = int(total_value - son_1_total_value)
    output = {
        "son_1_items": son_1_items,
        "son_1_total_value": son_1_total_value,
        "son_2_items": son_2_items,
        "son_2_total_value": son_2_total_value,
        "absolute_difference": int(round(abs(son_1_total_value - son_2_total_value)))
    }
    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": int(round(solver.Objective().Value())),
        "example_output": output
    }
else:
    output = {
        "son_1_items": [],
        "son_1_total_value": 0,
        "son_2_items": [],
        "son_2_total_value": 0,
        "absolute_difference": 0
    }
    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))