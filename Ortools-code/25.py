import json
from ortools.linear_solver import pywraplp

inp = {
    "initial_capital": 100000,
    "planning_horizon_years": 3,
    "investments": {
        "option_1": {
            "duration_years": 1,
            "maturity_value_per_yuan_invested": 1.7,
            "allowed_start_years": [1, 2, 3]
        },
        "option_2": {
            "duration_years": 2,
            "maturity_value_per_yuan_invested": 3.0,
            "allowed_start_years": [1, 2]
        }
    }
}

def build_model(data, fixed_final_wealth=None):
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("Failed to create OR-Tools solver.")

    H = data["planning_horizon_years"]
    initial_capital = data["initial_capital"]
    investments = data["investments"]
    INF = solver.infinity()

    x = {}
    for opt_name, opt_data in investments.items():
        x[opt_name] = {}
        for t in opt_data["allowed_start_years"]:
            x[opt_name][t] = solver.NumVar(0.0, INF, f"x_{opt_name}_{t}")

    cash = {t: solver.NumVar(0.0, INF, f"cash_{t}") for t in range(1, H + 2)}

    solver.Add(cash[1] == initial_capital)

    for year_end in range(1, H + 1):
        invest_now_terms = []
        mature_now_terms = []

        for opt_name, opt_data in investments.items():
            duration = opt_data["duration_years"]
            maturity_value = opt_data["maturity_value_per_yuan_invested"]

            if year_end in x[opt_name]:
                invest_now_terms.append(x[opt_name][year_end])

            for start_year in opt_data["allowed_start_years"]:
                if start_year + duration - 1 == year_end:
                    mature_now_terms.append(maturity_value * x[opt_name][start_year])

        invest_now = solver.Sum(invest_now_terms) if invest_now_terms else 0.0
        mature_now = solver.Sum(mature_now_terms) if mature_now_terms else 0.0

        solver.Add(invest_now <= cash[year_end])
        solver.Add(cash[year_end + 1] == cash[year_end] - invest_now + mature_now)

    if fixed_final_wealth is not None:
        solver.Add(cash[H + 1] == fixed_final_wealth)

    return solver, x, cash

def clean(v):
    if abs(v) < 1e-9:
        return 0.0
    return round(float(v), 6)

# Phase 1: maximize final wealth
solver1, x1, cash1 = build_model(inp)
H = inp["planning_horizon_years"]

obj1 = solver1.Objective()
obj1.SetCoefficient(cash1[H + 1], 1.0)
obj1.SetMaximization()

status1 = solver1.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

if status1 not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    result = {
        "status": status_map.get(status1, str(status1)),
        "objective_value": None,
        "example_output": None
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit

optimal_final_wealth = cash1[H + 1].solution_value()

# Phase 2: tie-break among optimal solutions to get a deterministic investment plan
# Prefer investing as much as possible in option 2 at year 1.
solver2, x2, cash2 = build_model(inp, fixed_final_wealth=optimal_final_wealth)
obj2 = solver2.Objective()
obj2.SetCoefficient(x2["option_2"][1], 1.0)
obj2.SetMaximization()

status2 = solver2.Solve()

if status2 not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    solver2, x2, cash2 = solver1, x1, cash1
    final_status = status1
else:
    final_status = status2

output = {
    "investment_plan": {
        "option_1_by_start_year": {
            str(t): clean(x2["option_1"][t].solution_value())
            for t in inp["investments"]["option_1"]["allowed_start_years"]
        },
        "option_2_by_start_year": {
            str(t): clean(x2["option_2"][t].solution_value())
            for t in inp["investments"]["option_2"]["allowed_start_years"]
        }
    },
    "cash_available_by_start_of_year": {
        str(t): clean(cash2[t].solution_value())
        for t in range(1, H + 1)
    },
    "final_wealth_end_of_year_3": clean(cash2[H + 1].solution_value()),
    "total_earnings": clean(cash2[H + 1].solution_value() - inp["initial_capital"])
}

result = {
    "status": status_map.get(final_status, str(final_status)),
    "objective_value": clean(cash2[H + 1].solution_value()),
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False, indent=2))