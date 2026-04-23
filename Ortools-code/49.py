import json
from ortools.linear_solver import pywraplp

inp = {
    "months": [1, 2, 3, 4],
    "required_area_sqm": {
        "1": 1500,
        "2": 1000,
        "3": 2000,
        "4": 1200
    },
    "contract_options": [
        {
            "length_months": 1,
            "rental_fee_per_100sqm_yuan": 4000
        },
        {
            "length_months": 2,
            "rental_fee_per_100sqm_yuan": 7500
        },
        {
            "length_months": 3,
            "rental_fee_per_100sqm_yuan": 10500
        },
        {
            "length_months": 4,
            "rental_fee_per_100sqm_yuan": 13000
        }
    ],
    "area_unit_sqm": 100,
    "rules": {
        "allow_parallel_contracts": True,
        "min_distinct_contract_lengths": 2,
        "max_distinct_contract_lengths": 3,
        "mutual_exclusion_lengths": [1, 4],
        "exact_monthly_satisfaction": True
    }
}

months = inp["months"]
n_months = len(months)
area_unit = inp["area_unit_sqm"]

required_area_sqm = {int(k): int(v) for k, v in inp["required_area_sqm"].items()}
for m in months:
    if required_area_sqm[m] % area_unit != 0:
        raise ValueError("Required area must be divisible by area_unit_sqm.")
required_units = {m: required_area_sqm[m] // area_unit for m in months}

contract_lengths = [opt["length_months"] for opt in inp["contract_options"]]
cost_per_unit = {
    opt["length_months"]: opt["rental_fee_per_100sqm_yuan"]
    for opt in inp["contract_options"]
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

# x[s, l] = number of 100-sqm contracts that start in month s and last l months
feasible_pairs = [
    (s, l)
    for s in months
    for l in contract_lengths
    if s + l - 1 <= n_months
]

x = {
    (s, l): solver.IntVar(0, solver.infinity(), f"x_{s}_{l}")
    for (s, l) in feasible_pairs
}

# y[l] = 1 if any contract with length l is used
y = {l: solver.BoolVar(f"y_{l}") for l in contract_lengths}

# Exact monthly demand satisfaction
for m in months:
    solver.Add(
        sum(x[s, l] for (s, l) in feasible_pairs if s <= m <= s + l - 1) == required_units[m]
    )

# Link x and y
big_m = sum(required_units.values())
for l in contract_lengths:
    x_of_length_l = [x[s, ll] for (s, ll) in feasible_pairs if ll == l]
    if x_of_length_l:
        solver.Add(sum(x_of_length_l) <= big_m * y[l])
        solver.Add(sum(x_of_length_l) >= y[l])

# Business rules on distinct lengths
solver.Add(sum(y[l] for l in contract_lengths) >= inp["rules"]["min_distinct_contract_lengths"])
solver.Add(sum(y[l] for l in contract_lengths) <= inp["rules"]["max_distinct_contract_lengths"])

a, b = inp["rules"]["mutual_exclusion_lengths"]
solver.Add(y[a] + y[b] <= 1)

# Objective: minimize total rental cost
objective = solver.Objective()
for (s, l), var in x.items():
    objective.SetCoefficient(var, cost_per_unit[l])
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
status = status_map.get(status_code, str(status_code))

objective_value = None
output = {
    "minimum_total_rental_cost_yuan": None,
    "used_contract_lengths_months": [],
    "contracts": [],
    "monthly_covered_area_sqm": {str(m): 0 for m in months}
}

if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    objective_value = int(round(objective.Value()))

    used_lengths = sorted(
        l for l in contract_lengths if sum(int(round(x[s, ll].solution_value())) for (s, ll) in feasible_pairs if ll == l) > 0
    )

    contracts = []
    for s, l in sorted(feasible_pairs):
        units = int(round(x[s, l].solution_value()))
        if units > 0:
            contracts.append({
                "start_month": s,
                "length_months": l,
                "units_of_100sqm": units,
                "area_sqm": units * area_unit,
                "cost_yuan": units * cost_per_unit[l]
            })

    monthly_covered_area_sqm = {}
    for m in months:
        covered_units = sum(
            int(round(x[s, l].solution_value()))
            for (s, l) in feasible_pairs
            if s <= m <= s + l - 1
        )
        monthly_covered_area_sqm[str(m)] = covered_units * area_unit

    output = {
        "minimum_total_rental_cost_yuan": objective_value,
        "used_contract_lengths_months": used_lengths,
        "contracts": contracts,
        "monthly_covered_area_sqm": monthly_covered_area_sqm
    }

print(json.dumps({
    "status": status,
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))