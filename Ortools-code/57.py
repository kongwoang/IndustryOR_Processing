import json
from itertools import product
from ortools.linear_solver import pywraplp

inp = {
    "orders": [
        {"order_number": 1, "width": 0.5, "length": 1000.0},
        {"order_number": 2, "width": 0.7, "length": 3000.0},
        {"order_number": 3, "width": 0.9, "length": 2000.0},
    ],
    "standard_roll_widths": [1.0, 2.0],
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("Could not create GLOP solver.")

order_widths = [o["width"] for o in inp["orders"]]
demands = {o["width"]: o["length"] for o in inp["orders"]}
min_width = min(order_widths)

# Generate maximal cutting patterns for each stock width:
# a pattern is kept if it fits and the leftover is smaller than the smallest order width,
# meaning no additional strip can still be cut from the leftover width.
patterns = []
for stock_width in inp["standard_roll_widths"]:
    max_counts = [int(stock_width // w) for w in order_widths]
    seen = set()
    for counts in product(*[range(m + 1) for m in max_counts]):
        if sum(counts) == 0:
            continue
        used_width = sum(c * w for c, w in zip(counts, order_widths))
        if used_width <= stock_width + 1e-9:
            leftover = stock_width - used_width
            if leftover < min_width - 1e-9:
                key = (round(stock_width, 10), counts)
                if key not in seen:
                    seen.add(key)
                    patterns.append({
                        "stock_width": stock_width,
                        "counts": counts,
                        "waste_width_per_meter": leftover,
                    })

# Decision variables: length (meters) assigned to each pattern
x = []
for i, p in enumerate(patterns):
    x.append(solver.NumVar(0.0, solver.infinity(), f"x_{i}"))

# Exact fulfillment of required lengths for each ordered width
for width in order_widths:
    ct = solver.Constraint(demands[width], demands[width])
    for var, p in zip(x, patterns):
        idx = order_widths.index(width)
        ct.SetCoefficient(var, p["counts"][idx])

# Objective: minimize waste area = waste_width_per_meter * pattern_length
objective = solver.Objective()
for var, p in zip(x, patterns):
    objective.SetCoefficient(var, p["waste_width_per_meter"])
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
status = status_map.get(status_code, "UNKNOWN")

patterns_used = []
if status in {"OPTIMAL", "FEASIBLE"}:
    for var, p in zip(x, patterns):
        length_used = var.solution_value()
        if length_used > 1e-7:
            cuts = []
            for width, count in zip(order_widths, p["counts"]):
                if count > 0:
                    cuts.append({
                        "width": float(width),
                        "count_per_meter": int(count)
                    })
            waste_area = p["waste_width_per_meter"] * length_used
            patterns_used.append({
                "stock_width": float(p["stock_width"]),
                "cuts": cuts,
                "length_used": round(length_used, 6),
                "waste_width_per_meter": round(p["waste_width_per_meter"], 6),
                "waste_area": round(waste_area, 6),
            })

    patterns_used.sort(
        key=lambda item: (
            item["stock_width"],
            tuple((c["width"], c["count_per_meter"]) for c in item["cuts"])
        )
    )
    total_waste_area = round(solver.Objective().Value(), 6)
else:
    total_waste_area = None

output = {
    "total_waste_area": total_waste_area,
    "patterns_used": patterns_used
}

print(json.dumps({
    "status": status,
    "objective_value": total_waste_area,
    "example_output": output
}, ensure_ascii=False))