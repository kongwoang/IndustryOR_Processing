import json
from ortools.linear_solver import pywraplp

inp = {
    "stock_length": 10,
    "orders": [
        {
            "piece_length": 3,
            "required_quantity": 90
        },
        {
            "piece_length": 4,
            "required_quantity": 60
        }
    ]
}

stock_length = inp["stock_length"]
piece_lengths = [o["piece_length"] for o in inp["orders"]]
demands = {o["piece_length"]: o["required_quantity"] for o in inp["orders"]}

# Enumerate all feasible cutting patterns for one 10m raw bar.
patterns = []
max_a = stock_length // piece_lengths[0]
max_b = stock_length // piece_lengths[1]
for a in range(max_a + 1):
    for b in range(max_b + 1):
        used = a * piece_lengths[0] + b * piece_lengths[1]
        if used <= stock_length and (a > 0 or b > 0):
            patterns.append({
                "piece_counts": {
                    str(piece_lengths[0]): a,
                    str(piece_lengths[1]): b
                },
                "waste_per_bar": stock_length - used
            })

# Sort for stable output.
patterns.sort(
    key=lambda p: (
        p["waste_per_bar"],
        -p["piece_counts"][str(piece_lengths[0])],
        -p["piece_counts"][str(piece_lengths[1])]
    )
)

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

x = []
for i, p in enumerate(patterns):
    x.append(solver.IntVar(0.0, solver.infinity(), f"x_{i}"))

# Demand satisfaction constraints.
for L in piece_lengths:
    solver.Add(
        sum(x[i] * patterns[i]["piece_counts"][str(L)] for i in range(len(patterns))) == demands[L]
    )

# Minimize total waste.
objective = solver.Objective()
for i, p in enumerate(patterns):
    objective.SetCoefficient(x[i], p["waste_per_bar"])
objective.SetMinimization()

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

output = {
    "total_raw_bars_used": 0,
    "total_waste": None,
    "patterns_used": []
}

objective_value = None

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    total_bars = 0
    total_waste = 0.0
    for i, p in enumerate(patterns):
        bars = int(round(x[i].solution_value()))
        if bars > 0:
            pattern_total_waste = bars * p["waste_per_bar"]
            output["patterns_used"].append({
                "piece_counts": p["piece_counts"],
                "bars": bars,
                "waste_per_bar": p["waste_per_bar"],
                "total_waste": pattern_total_waste
            })
            total_bars += bars
            total_waste += pattern_total_waste

    output["total_raw_bars_used"] = total_bars
    output["total_waste"] = total_waste
    objective_value = solver.Objective().Value()

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))