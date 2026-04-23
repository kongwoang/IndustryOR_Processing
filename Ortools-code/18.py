import json
from ortools.linear_solver import pywraplp

# Source: :contentReference[oaicite:0]{index=0}

inp = {
    "regular_hours_per_week": 110,
    "overtime_limit_hours_per_week": 10,
    "products": [
        {
            "name": "curtain",
            "production_rate_meters_per_hour": 1000,
            "minimum_weekly_sales_meters": 70000,
            "profit_per_meter_yuan": 2.5
        },
        {
            "name": "clothing",
            "production_rate_meters_per_hour": 1000,
            "minimum_weekly_sales_meters": 45000,
            "profit_per_meter_yuan": 1.5
        }
    ],
    "priority_order": [
        "fully_use_regular_time",
        "keep_overtime_within_10_hours",
        "meet_minimum_weekly_sales",
        "minimize_overtime"
    ]
}

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("GLOP solver is not available.")

inf = solver.infinity()

products = {p["name"]: p for p in inp["products"]}
curtain = products["curtain"]
clothing = products["clothing"]

# Decision variables
x_curtain = solver.NumVar(0.0, inf, "x_curtain")
x_clothing = solver.NumVar(0.0, inf, "x_clothing")
regular_hours_used = solver.NumVar(0.0, inp["regular_hours_per_week"], "regular_hours_used")
overtime_hours = solver.NumVar(0.0, inf, "overtime_hours")

# Deviation variables for preemptive goal programming
d1_unused_regular = solver.NumVar(0.0, inf, "d1_unused_regular")
d2_under_overtime_limit = solver.NumVar(0.0, inf, "d2_under_overtime_limit")
d2_excess_overtime = solver.NumVar(0.0, inf, "d2_excess_overtime")
d3_curtain_shortfall = solver.NumVar(0.0, inf, "d3_curtain_shortfall")
d3_curtain_surplus = solver.NumVar(0.0, inf, "d3_curtain_surplus")
d3_clothing_shortfall = solver.NumVar(0.0, inf, "d3_clothing_shortfall")
d3_clothing_surplus = solver.NumVar(0.0, inf, "d3_clothing_surplus")

# Core balance constraint: production time equals regular time plus overtime
time_balance = solver.Constraint(0.0, 0.0, "time_balance")
time_balance.SetCoefficient(x_curtain, 1.0 / curtain["production_rate_meters_per_hour"])
time_balance.SetCoefficient(x_clothing, 1.0 / clothing["production_rate_meters_per_hour"])
time_balance.SetCoefficient(regular_hours_used, -1.0)
time_balance.SetCoefficient(overtime_hours, -1.0)

# Goal equations
# p1: fully utilize 110 regular hours
c_p1 = solver.Constraint(inp["regular_hours_per_week"], inp["regular_hours_per_week"], "goal_p1")
c_p1.SetCoefficient(regular_hours_used, 1.0)
c_p1.SetCoefficient(d1_unused_regular, 1.0)

# p2: overtime should not exceed 10 hours
c_p2 = solver.Constraint(inp["overtime_limit_hours_per_week"], inp["overtime_limit_hours_per_week"], "goal_p2")
c_p2.SetCoefficient(overtime_hours, 1.0)
c_p2.SetCoefficient(d2_under_overtime_limit, 1.0)
c_p2.SetCoefficient(d2_excess_overtime, -1.0)

# p3: minimum weekly sales targets
c_p3_curtain = solver.Constraint(curtain["minimum_weekly_sales_meters"], curtain["minimum_weekly_sales_meters"], "goal_p3_curtain")
c_p3_curtain.SetCoefficient(x_curtain, 1.0)
c_p3_curtain.SetCoefficient(d3_curtain_shortfall, 1.0)
c_p3_curtain.SetCoefficient(d3_curtain_surplus, -1.0)

c_p3_clothing = solver.Constraint(clothing["minimum_weekly_sales_meters"], clothing["minimum_weekly_sales_meters"], "goal_p3_clothing")
c_p3_clothing.SetCoefficient(x_clothing, 1.0)
c_p3_clothing.SetCoefficient(d3_clothing_shortfall, 1.0)
c_p3_clothing.SetCoefficient(d3_clothing_surplus, -1.0)

def solve_minimize(linear_terms):
    objective = solver.Objective()
    objective.Clear()
    for var, coef in linear_terms:
        objective.SetCoefficient(var, coef)
    objective.SetMinimization()
    return solver.Solve()

def fix_value(var, value, tol=1e-7):
    c1 = solver.Constraint(-inf, value + tol, f"fix_ub_{var.name()}_{solver.NumConstraints()}")
    c1.SetCoefficient(var, 1.0)
    c2 = solver.Constraint(value - tol, inf, f"fix_lb_{var.name()}_{solver.NumConstraints()}")
    c2.SetCoefficient(var, 1.0)

# Priority 1: minimize unused regular hours
status = solve_minimize([(d1_unused_regular, 1.0)])
if status != pywraplp.Solver.OPTIMAL:
    raise RuntimeError("Failed at priority 1.")
p1_value = d1_unused_regular.solution_value()
fix_value(d1_unused_regular, p1_value)

# Priority 2: minimize overtime excess over 10 hours
status = solve_minimize([(d2_excess_overtime, 1.0)])
if status != pywraplp.Solver.OPTIMAL:
    raise RuntimeError("Failed at priority 2.")
p2_value = d2_excess_overtime.solution_value()
fix_value(d2_excess_overtime, p2_value)

# Priority 3: minimize total shortfall in meeting minimum sales
status = solve_minimize([
    (d3_curtain_shortfall, 1.0),
    (d3_clothing_shortfall, 1.0)
])
if status != pywraplp.Solver.OPTIMAL:
    raise RuntimeError("Failed at priority 3.")
p3_value = d3_curtain_shortfall.solution_value() + d3_clothing_shortfall.solution_value()
fix_value(d3_curtain_shortfall, d3_curtain_shortfall.solution_value())
fix_value(d3_clothing_shortfall, d3_clothing_shortfall.solution_value())

# Priority 4: minimize overtime
status = solve_minimize([(overtime_hours, 1.0)])
if status != pywraplp.Solver.OPTIMAL:
    raise RuntimeError("Failed at priority 4.")
p4_value = overtime_hours.solution_value()

def clean_number(x, tol=1e-6):
    if abs(x - round(x)) <= tol:
        return int(round(x))
    return float(round(x, 6))

profit = (
    curtain["profit_per_meter_yuan"] * x_curtain.solution_value() +
    clothing["profit_per_meter_yuan"] * x_clothing.solution_value()
)

output = {
    "production_meters": {
        "curtain": clean_number(x_curtain.solution_value()),
        "clothing": clean_number(x_clothing.solution_value())
    },
    "hours": {
        "regular": clean_number(regular_hours_used.solution_value()),
        "overtime": clean_number(overtime_hours.solution_value()),
        "total": clean_number(regular_hours_used.solution_value() + overtime_hours.solution_value())
    },
    "profit_yuan": clean_number(profit),
    "goal_achievement": {
        "unused_regular_hours": clean_number(d1_unused_regular.solution_value()),
        "overtime_excess_over_10_hours": clean_number(d2_excess_overtime.solution_value()),
        "curtain_sales_shortfall_meters": clean_number(d3_curtain_shortfall.solution_value()),
        "clothing_sales_shortfall_meters": clean_number(d3_clothing_shortfall.solution_value())
    }
}

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

objective_value = clean_number(overtime_hours.solution_value())

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))