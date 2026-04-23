import json
from ortools.linear_solver import pywraplp

inp = {
    "selling_prices_yuan_per_t": {
        "type_I": 4800,
        "type_II": 5600
    },
    "min_proportion_A": {
        "type_I": 0.5,
        "type_II": 0.6
    },
    "inventory_t": {
        "crude_A": 500,
        "crude_B": 1000
    },
    "purchase_tiers_for_A": [
        {
            "name": "tier_1_up_to_500",
            "max_qty_t": 500,
            "unit_price_yuan_per_t": 10000
        },
        {
            "name": "tier_2_500_to_1000",
            "max_qty_t": 500,
            "unit_price_yuan_per_t": 8000
        },
        {
            "name": "tier_3_1000_to_1500",
            "max_qty_t": 500,
            "unit_price_yuan_per_t": 6000
        }
    ]
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    solver = pywraplp.Solver.CreateSolver("CBC")
if solver is None:
    raise RuntimeError("No suitable MIP solver available.")

# Blend allocation variables (tons of crude sent to each gasoline type)
a1 = solver.NumVar(0.0, solver.infinity(), "A_to_type_I")
b1 = solver.NumVar(0.0, solver.infinity(), "B_to_type_I")
a2 = solver.NumVar(0.0, solver.infinity(), "A_to_type_II")
b2 = solver.NumVar(0.0, solver.infinity(), "B_to_type_II")

# Purchase variables by price tier
tier_caps = [t["max_qty_t"] for t in inp["purchase_tiers_for_A"]]
tier_prices = [t["unit_price_yuan_per_t"] for t in inp["purchase_tiers_for_A"]]
tier_names = [t["name"] for t in inp["purchase_tiers_for_A"]]

p1 = solver.NumVar(0.0, tier_caps[0], "purchase_tier_1")
p2 = solver.NumVar(0.0, tier_caps[1], "purchase_tier_2")
p3 = solver.NumVar(0.0, tier_caps[2], "purchase_tier_3")

# Binary variables to enforce sequential use of purchase tiers
z1 = solver.IntVar(0, 1, "z1")
z2 = solver.IntVar(0, 1, "z2")
z3 = solver.IntVar(0, 1, "z3")

# Tier activation / sequencing constraints
solver.Add(p1 <= tier_caps[0] * z1)
solver.Add(p2 <= tier_caps[1] * z2)
solver.Add(p3 <= tier_caps[2] * z3)
solver.Add(z1 >= z2)
solver.Add(z2 >= z3)
solver.Add(p1 >= tier_caps[0] * z2)  # if tier 2 is used, tier 1 must be full
solver.Add(p2 >= tier_caps[1] * z3)  # if tier 3 is used, tier 2 must be full

total_purchase_A = p1 + p2 + p3

# Inventory constraints
solver.Add(a1 + a2 <= inp["inventory_t"]["crude_A"] + total_purchase_A)
solver.Add(b1 + b2 <= inp["inventory_t"]["crude_B"])

# Minimum proportion constraints
# Type I: A / (A + B) >= 0.5  => A >= B
solver.Add(a1 >= b1)
# Type II: A / (A + B) >= 0.6 => 2A >= 3B
solver.Add(2 * a2 >= 3 * b2)

# Objective: maximize sales revenue minus crude A purchase cost
revenue = (
    inp["selling_prices_yuan_per_t"]["type_I"] * (a1 + b1) +
    inp["selling_prices_yuan_per_t"]["type_II"] * (a2 + b2)
)
purchase_cost = tier_prices[0] * p1 + tier_prices[1] * p2 + tier_prices[2] * p3
solver.Maximize(revenue - purchase_cost)

status = solver.Solve()

def clean(x):
    if abs(x) < 1e-9:
        return 0
    rx = round(x)
    if abs(x - rx) < 1e-9:
        return int(rx)
    return round(x, 6)

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    output = {
        "purchase_A_by_tier_t": {
            tier_names[0]: clean(p1.solution_value()),
            tier_names[1]: clean(p2.solution_value()),
            tier_names[2]: clean(p3.solution_value())
        },
        "total_purchase_A_t": clean(total_purchase_A.solution_value()),
        "allocation_t": {
            "A_to_type_I": clean(a1.solution_value()),
            "B_to_type_I": clean(b1.solution_value()),
            "A_to_type_II": clean(a2.solution_value()),
            "B_to_type_II": clean(b2.solution_value())
        },
        "gasoline_output_t": {
            "type_I": clean(a1.solution_value() + b1.solution_value()),
            "type_II": clean(a2.solution_value() + b2.solution_value())
        }
    }
    result = {
        "status": "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE",
        "objective_value": clean(solver.Objective().Value()),
        "example_output": output
    }
else:
    output = {
        "purchase_A_by_tier_t": {
            "tier_1_up_to_500": None,
            "tier_2_500_to_1000": None,
            "tier_3_1000_to_1500": None
        },
        "total_purchase_A_t": None,
        "allocation_t": {
            "A_to_type_I": None,
            "B_to_type_I": None,
            "A_to_type_II": None,
            "B_to_type_II": None
        },
        "gasoline_output_t": {
            "type_I": None,
            "type_II": None
        }
    }
    status_map = {
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
    }
    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))