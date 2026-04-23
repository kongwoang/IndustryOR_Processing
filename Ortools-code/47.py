import json
from ortools.linear_solver import pywraplp

inp = {
    "resources": {
        "land_hectares": 100,
        "development_funds_yuan": 15000,
        "labor_person_days": {
            "autumn_winter": 3500,
            "spring_summer": 4000
        },
        "external_work_wage_yuan_per_person_day": {
            "autumn_winter": 1.8,
            "spring_summer": 2.1
        }
    },
    "crops": {
        "soybean": {
            "autumn_winter_person_days_per_hectare": 20,
            "spring_summer_person_days_per_hectare": 50,
            "annual_net_income_yuan_per_hectare": 175
        },
        "corn": {
            "autumn_winter_person_days_per_hectare": 35,
            "spring_summer_person_days_per_hectare": 75,
            "annual_net_income_yuan_per_hectare": 300
        },
        "wheat": {
            "autumn_winter_person_days_per_hectare": 10,
            "spring_summer_person_days_per_hectare": 40,
            "annual_net_income_yuan_per_hectare": 120
        }
    },
    "livestock": {
        "dairy_cow": {
            "investment_yuan_per_head": 400,
            "land_hectares_per_head": 1.5,
            "autumn_winter_person_days_per_head": 100,
            "spring_summer_person_days_per_head": 50,
            "annual_net_income_yuan_per_head": 400,
            "max_heads": 32
        },
        "chicken": {
            "investment_yuan_per_head": 3,
            "land_hectares_per_head": 0,
            "autumn_winter_person_days_per_head": 0.6,
            "spring_summer_person_days_per_head": 0.3,
            "annual_net_income_yuan_per_head": 2,
            "max_heads": 3000
        }
    },
    "integrality": {
        "labor_days_must_be_whole_numbers": True
    }
}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is unavailable.")

R = inp["resources"]
C = inp["crops"]
L = inp["livestock"]

# Crop areas should remain continuous.
soy = solver.NumVar(0.0, R["land_hectares"], "soy_hectares")
corn = solver.NumVar(0.0, R["land_hectares"], "corn_hectares")
wheat = solver.NumVar(0.0, R["land_hectares"], "wheat_hectares")

# Animal counts are integer.
cows = solver.IntVar(0, L["dairy_cow"]["max_heads"], "dairy_cows")
chickens = solver.IntVar(0, L["chicken"]["max_heads"], "chickens")

# Unused labor can be allocated to external work.
ext_aw = solver.NumVar(0.0, R["labor_person_days"]["autumn_winter"], "external_aw_days")
ext_ss = solver.NumVar(0.0, R["labor_person_days"]["spring_summer"], "external_ss_days")

# Land
solver.Add(
    soy + corn + wheat + L["dairy_cow"]["land_hectares_per_head"] * cows
    <= R["land_hectares"]
)

# Development funds
solver.Add(
    L["dairy_cow"]["investment_yuan_per_head"] * cows
    + L["chicken"]["investment_yuan_per_head"] * chickens
    <= R["development_funds_yuan"]
)

# Labor balances
solver.Add(
    C["soybean"]["autumn_winter_person_days_per_hectare"] * soy
    + C["corn"]["autumn_winter_person_days_per_hectare"] * corn
    + C["wheat"]["autumn_winter_person_days_per_hectare"] * wheat
    + L["dairy_cow"]["autumn_winter_person_days_per_head"] * cows
    + L["chicken"]["autumn_winter_person_days_per_head"] * chickens
    + ext_aw
    == R["labor_person_days"]["autumn_winter"]
)

solver.Add(
    C["soybean"]["spring_summer_person_days_per_hectare"] * soy
    + C["corn"]["spring_summer_person_days_per_hectare"] * corn
    + C["wheat"]["spring_summer_person_days_per_hectare"] * wheat
    + L["dairy_cow"]["spring_summer_person_days_per_head"] * cows
    + L["chicken"]["spring_summer_person_days_per_head"] * chickens
    + ext_ss
    == R["labor_person_days"]["spring_summer"]
)

# Objective: farm income + external labor income
objective = solver.Objective()
objective.SetCoefficient(soy, C["soybean"]["annual_net_income_yuan_per_hectare"])
objective.SetCoefficient(corn, C["corn"]["annual_net_income_yuan_per_hectare"])
objective.SetCoefficient(wheat, C["wheat"]["annual_net_income_yuan_per_hectare"])
objective.SetCoefficient(cows, L["dairy_cow"]["annual_net_income_yuan_per_head"])
objective.SetCoefficient(chickens, L["chicken"]["annual_net_income_yuan_per_head"])
objective.SetCoefficient(ext_aw, R["external_work_wage_yuan_per_person_day"]["autumn_winter"])
objective.SetCoefficient(ext_ss, R["external_work_wage_yuan_per_person_day"]["spring_summer"])
objective.SetMaximization()

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

def clean_number(x):
    if x is None:
        return None
    if abs(x - round(x)) <= 1e-9:
        return int(round(x))
    return float(x)

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    soy_val = soy.solution_value()
    corn_val = corn.solution_value()
    wheat_val = wheat.solution_value()
    cows_val = cows.solution_value()
    chickens_val = chickens.solution_value()
    ext_aw_val = ext_aw.solution_value()
    ext_ss_val = ext_ss.solution_value()

    land_used = soy_val + corn_val + wheat_val + L["dairy_cow"]["land_hectares_per_head"] * cows_val
    funds_used = (
        L["dairy_cow"]["investment_yuan_per_head"] * cows_val
        + L["chicken"]["investment_yuan_per_head"] * chickens_val
    )
    aw_used = (
        C["soybean"]["autumn_winter_person_days_per_hectare"] * soy_val
        + C["corn"]["autumn_winter_person_days_per_hectare"] * corn_val
        + C["wheat"]["autumn_winter_person_days_per_hectare"] * wheat_val
        + L["dairy_cow"]["autumn_winter_person_days_per_head"] * cows_val
        + L["chicken"]["autumn_winter_person_days_per_head"] * chickens_val
    )
    ss_used = (
        C["soybean"]["spring_summer_person_days_per_hectare"] * soy_val
        + C["corn"]["spring_summer_person_days_per_hectare"] * corn_val
        + C["wheat"]["spring_summer_person_days_per_hectare"] * wheat_val
        + L["dairy_cow"]["spring_summer_person_days_per_head"] * cows_val
        + L["chicken"]["spring_summer_person_days_per_head"] * chickens_val
    )

    objective_value = solver.Objective().Value()
    output = {
        "crop_plan_hectares": {
            "soybean": clean_number(soy_val),
            "corn": clean_number(corn_val),
            "wheat": clean_number(wheat_val)
        },
        "livestock_plan": {
            "dairy_cows": clean_number(cows_val),
            "chickens": clean_number(chickens_val)
        },
        "external_work_person_days": {
            "autumn_winter": clean_number(ext_aw_val),
            "spring_summer": clean_number(ext_ss_val)
        },
        "resource_usage": {
            "land_hectares_used": clean_number(land_used),
            "funds_yuan_used": clean_number(funds_used),
            "autumn_winter_person_days_used": clean_number(aw_used),
            "spring_summer_person_days_used": clean_number(ss_used)
        },
        "total_annual_net_income_yuan": clean_number(objective_value)
    }
else:
    objective_value = None
    output = {
        "crop_plan_hectares": {
            "soybean": None,
            "corn": None,
            "wheat": None
        },
        "livestock_plan": {
            "dairy_cows": None,
            "chickens": None
        },
        "external_work_person_days": {
            "autumn_winter": None,
            "spring_summer": None
        },
        "resource_usage": {
            "land_hectares_used": None,
            "funds_yuan_used": None,
            "autumn_winter_person_days_used": None,
            "spring_summer_person_days_used": None
        },
        "total_annual_net_income_yuan": None
    }

print(json.dumps({
    "status": status_map.get(status, str(status)),
    "objective_value": clean_number(objective_value),
    "example_output": output
}, ensure_ascii=False))