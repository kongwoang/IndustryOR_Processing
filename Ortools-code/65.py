import json
from ortools.linear_solver import pywraplp

inp = json.loads(
    r'''
{
  "products": {
    "microwave_ovens": {
      "workshop_a_hours_per_unit": 2,
      "workshop_b_hours_per_unit": 1,
      "inspection_and_sales_cost_per_unit": 30
    },
    "water_heaters": {
      "workshop_a_hours_per_unit": 1,
      "workshop_b_hours_per_unit": 3,
      "inspection_and_sales_cost_per_unit": 50
    }
  },
  "workshops": {
    "A": {
      "regular_hours_available": 250,
      "hourly_cost": 80,
      "max_overtime_hours": 20
    },
    "B": {
      "regular_hours_available": 150,
      "hourly_cost": 20
    }
  },
  "constraints": {
    "max_monthly_inspection_and_sales_cost": 5500,
    "minimum_monthly_sales": {
      "microwave_ovens": 80,
      "water_heaters": 50
    }
  }
}
'''
)

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

inf = solver.infinity()

mw = inp["products"]["microwave_ovens"]
wh = inp["products"]["water_heaters"]
A = inp["workshops"]["A"]
B = inp["workshops"]["B"]
cons = inp["constraints"]

x = solver.IntVar(0, inf, "microwave_ovens")
y = solver.IntVar(0, inf, "water_heaters")

# Demand lower bounds
solver.Add(x >= cons["minimum_monthly_sales"]["microwave_ovens"])
solver.Add(y >= cons["minimum_monthly_sales"]["water_heaters"])

# Inspection and sales budget
solver.Add(
    mw["inspection_and_sales_cost_per_unit"] * x
    + wh["inspection_and_sales_cost_per_unit"] * y
    <= cons["max_monthly_inspection_and_sales_cost"]
)

# Workshop A regular time must be fully used, with at most 20 overtime hours
solver.Add(
    mw["workshop_a_hours_per_unit"] * x + wh["workshop_a_hours_per_unit"] * y
    >= A["regular_hours_available"]
)
solver.Add(
    mw["workshop_a_hours_per_unit"] * x + wh["workshop_a_hours_per_unit"] * y
    <= A["regular_hours_available"] + A["max_overtime_hours"]
)

# Workshop B available time should be at least fully utilized
solver.Add(
    mw["workshop_b_hours_per_unit"] * x + wh["workshop_b_hours_per_unit"] * y
    >= B["regular_hours_available"]
)

# Minimize total operating cost
objective = solver.Objective()
objective.SetCoefficient(
    x,
    mw["workshop_a_hours_per_unit"] * A["hourly_cost"]
    + mw["workshop_b_hours_per_unit"] * B["hourly_cost"]
    + mw["inspection_and_sales_cost_per_unit"],
)
objective.SetCoefficient(
    y,
    wh["workshop_a_hours_per_unit"] * A["hourly_cost"]
    + wh["workshop_b_hours_per_unit"] * B["hourly_cost"]
    + wh["inspection_and_sales_cost_per_unit"],
)
objective.SetMinimization()

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.MODEL_INVALID: "MODEL_INVALID",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    raise RuntimeError(f"Unexpected solve status: {status_map.get(status, str(status))}")

microwave_ovens = int(round(x.solution_value()))
water_heaters = int(round(y.solution_value()))

a_total_hours = (
    mw["workshop_a_hours_per_unit"] * microwave_ovens
    + wh["workshop_a_hours_per_unit"] * water_heaters
)
a_overtime_hours = max(0, a_total_hours - A["regular_hours_available"])
b_total_hours = (
    mw["workshop_b_hours_per_unit"] * microwave_ovens
    + wh["workshop_b_hours_per_unit"] * water_heaters
)
inspection_and_sales_cost = (
    mw["inspection_and_sales_cost_per_unit"] * microwave_ovens
    + wh["inspection_and_sales_cost_per_unit"] * water_heaters
)
total_operating_cost = (
    A["hourly_cost"] * a_total_hours
    + B["hourly_cost"] * b_total_hours
    + inspection_and_sales_cost
)

output = {
    "production_plan": {
        "microwave_ovens": microwave_ovens,
        "water_heaters": water_heaters
    },
    "workshop_hours": {
        "A_total_hours_used": a_total_hours,
        "A_overtime_hours": a_overtime_hours,
        "B_total_hours_used": b_total_hours
    },
    "inspection_and_sales_cost": inspection_and_sales_cost,
    "total_operating_cost": total_operating_cost
}

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": float(objective.Value()),
    "example_output": output
}, ensure_ascii=False))