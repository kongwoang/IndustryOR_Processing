import json
from ortools.linear_solver import pywraplp

inp = {
    "weeks": 8,
    "initial_skilled_workers": 50,
    "new_workers_to_train_by_end_of_week": 50,
    "hours_per_week": {
        "regular": 40,
        "overtime": 60
    },
    "training": {
        "duration_weeks": 2,
        "max_trainees_per_skilled_trainer": 3
    },
    "production_rate_kg_per_hour": {
        "I": 10,
        "II": 6
    },
    "weekly_wages_yuan": {
        "skilled_regular": 360,
        "skilled_overtime": 540,
        "trainee_during_training": 120,
        "trained_worker_after_training": 240
    },
    "delay_penalty_yuan_per_kg_per_week": {
        "I": 0.5,
        "II": 0.6
    },
    "demands_kg": {
        "I": [10000, 10000, 12000, 12000, 16000, 16000, 20000, 20000],
        "II": [6000, 7200, 8400, 10800, 10800, 12000, 12000, 12000]
    }
}

weeks = inp["weeks"]
initial_skilled = inp["initial_skilled_workers"]
target_new = inp["new_workers_to_train_by_end_of_week"]
regular_hours = inp["hours_per_week"]["regular"]
overtime_hours = inp["hours_per_week"]["overtime"]
training_duration = inp["training"]["duration_weeks"]
max_trainees_per_trainer = inp["training"]["max_trainees_per_skilled_trainer"]
rate_I = inp["production_rate_kg_per_hour"]["I"]
rate_II = inp["production_rate_kg_per_hour"]["II"]
wage_skilled_regular = inp["weekly_wages_yuan"]["skilled_regular"]
wage_skilled_overtime = inp["weekly_wages_yuan"]["skilled_overtime"]
wage_trainee = inp["weekly_wages_yuan"]["trainee_during_training"]
wage_trained = inp["weekly_wages_yuan"]["trained_worker_after_training"]
penalty_I = inp["delay_penalty_yuan_per_kg_per_week"]["I"]
penalty_II = inp["delay_penalty_yuan_per_kg_per_week"]["II"]
demand_I = inp["demands_kg"]["I"]
demand_II = inp["demands_kg"]["II"]

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    solver = pywraplp.Solver.CreateSolver("CBC")
if solver is None:
    raise RuntimeError("No suitable MILP solver available in OR-Tools.")

train_start_weeks = range(1, weeks)  # weeks 1..7 can start a 2-week training cycle
all_weeks = range(1, weeks + 1)

start = {t: solver.IntVar(0, target_new, f"start_{t}") for t in train_start_weeks}
trainers = {t: solver.IntVar(0, initial_skilled, f"trainers_{t}") for t in train_start_weeks}
ot = {t: solver.IntVar(0, initial_skilled, f"ot_{t}") for t in all_weeks}

prod_I = {t: solver.NumVar(0, solver.infinity(), f"prod_I_{t}") for t in all_weeks}
prod_II = {t: solver.NumVar(0, solver.infinity(), f"prod_II_{t}") for t in all_weeks}

inv_I = {t: solver.NumVar(0, solver.infinity(), f"inv_I_{t}") for t in all_weeks}
back_I = {t: solver.NumVar(0, solver.infinity(), f"back_I_{t}") for t in all_weeks}
inv_II = {t: solver.NumVar(0, solver.infinity(), f"inv_II_{t}") for t in all_weeks}
back_II = {t: solver.NumVar(0, solver.infinity(), f"back_II_{t}") for t in all_weeks}

def expr_sum(exprs):
    return sum(exprs) if exprs else 0

solver.Add(expr_sum(start[t] for t in train_start_weeks) == target_new)

for t in train_start_weeks:
    solver.Add(start[t] <= max_trainees_per_trainer * trainers[t])

for t in all_weeks:
    occupied_trainers = expr_sum(
        trainers[k] for k in train_start_weeks if k <= t <= k + training_duration - 1
    )
    trained_workers_available = expr_sum(
        start[k] for k in train_start_weeks if k + training_duration <= t
    )
    active_original_skilled = initial_skilled - occupied_trainers
    solver.Add(ot[t] <= active_original_skilled)

    total_available_hours = (
        regular_hours * active_original_skilled
        + (overtime_hours - regular_hours) * ot[t]
        + regular_hours * trained_workers_available
    )

    solver.Add(prod_I[t] * (1.0 / rate_I) + prod_II[t] * (1.0 / rate_II) <= total_available_hours)

for t in all_weeks:
    prev_net_I = 0 if t == 1 else inv_I[t - 1] - back_I[t - 1]
    prev_net_II = 0 if t == 1 else inv_II[t - 1] - back_II[t - 1]

    solver.Add(prev_net_I + prod_I[t] - demand_I[t - 1] == inv_I[t] - back_I[t])
    solver.Add(prev_net_II + prod_II[t] - demand_II[t - 1] == inv_II[t] - back_II[t])

solver.Add(inv_I[weeks] == 0)
solver.Add(back_I[weeks] == 0)
solver.Add(inv_II[weeks] == 0)
solver.Add(back_II[weeks] == 0)

objective = 0
for t in all_weeks:
    trainees_in_training = expr_sum(
        start[k] for k in train_start_weeks if k <= t <= k + training_duration - 1
    )
    trained_workers_available = expr_sum(
        start[k] for k in train_start_weeks if k + training_duration <= t
    )
    objective += (wage_skilled_overtime - wage_skilled_regular) * ot[t]
    objective += wage_trainee * trainees_in_training
    objective += wage_trained * trained_workers_available
    objective += penalty_I * back_I[t] + penalty_II * back_II[t]

solver.Minimize(objective)

status_code = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.MODEL_INVALID: "MODEL_INVALID",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

def clean_number(x):
    if abs(x) < 1e-9:
        return 0.0
    return round(float(x), 6)

output = {
    "training_starts": [],
    "weekly_plan": []
}

if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    for t in train_start_weeks:
        output["training_starts"].append({
            "week": t,
            "new_workers_started_training": int(round(start[t].solution_value())),
            "skilled_trainers_assigned": int(round(trainers[t].solution_value()))
        })

    for t in all_weeks:
        trained_workers_available_value = int(round(sum(
            start[k].solution_value() for k in train_start_weeks if k + training_duration <= t
        )))
        output["weekly_plan"].append({
            "week": t,
            "overtime_skilled_workers": int(round(ot[t].solution_value())),
            "trained_workers_available_for_production": trained_workers_available_value,
            "production_kg": {
                "I": clean_number(prod_I[t].solution_value()),
                "II": clean_number(prod_II[t].solution_value())
            },
            "ending_inventory_kg": {
                "I": clean_number(inv_I[t].solution_value()),
                "II": clean_number(inv_II[t].solution_value())
            },
            "ending_backlog_kg": {
                "I": clean_number(back_I[t].solution_value()),
                "II": clean_number(back_II[t].solution_value())
            }
        })

base_cost = weeks * initial_skilled * wage_skilled_regular
objective_value = None
if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    objective_value = clean_number(base_cost + solver.Objective().Value())

result = {
    "status": status_map.get(status_code, str(status_code)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))