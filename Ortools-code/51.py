# Source: :contentReference[oaicite:0]{index=0}
import json
from ortools.linear_solver import pywraplp

inp = {
    "time_periods": [
        {"start": "02:00", "end": "06:00", "required_nurses": 10},
        {"start": "06:00", "end": "10:00", "required_nurses": 15},
        {"start": "10:00", "end": "14:00", "required_nurses": 25},
        {"start": "14:00", "end": "18:00", "required_nurses": 20},
        {"start": "18:00", "end": "22:00", "required_nurses": 18},
        {"start": "22:00", "end": "02:00", "required_nurses": 12}
    ],
    "shift_start_times": ["02:00", "06:00", "10:00", "14:00", "18:00", "22:00"],
    "shift_length_hours": 8,
    "nurse_types": {
        "regular": {"hourly_pay": 10},
        "contract": {"hourly_pay": 15}
    }
}

def to_minutes(hhmm):
    h, m = map(int, hhmm.split(":"))
    return 60 * h + m

def duration_hours(start, end):
    s = to_minutes(start)
    e = to_minutes(end)
    d = (e - s) % (24 * 60)
    if d == 0:
        d = 24 * 60
    return d // 60

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    solver = pywraplp.Solver.CreateSolver("CBC")

starts = inp["shift_start_times"]
periods = inp["time_periods"]
n = len(starts)

period_len_hours = duration_hours(periods[0]["start"], periods[0]["end"])
coverage_blocks = inp["shift_length_hours"] // period_len_hours

regular = {s: solver.IntVar(0, solver.infinity(), f"regular_{s.replace(':', '')}") for s in starts}
contract = {s: solver.IntVar(0, solver.infinity(), f"contract_{s.replace(':', '')}") for s in starts}

for i, period in enumerate(periods):
    covered_expr = []
    for k in range(coverage_blocks):
        start_idx = (i - k) % n
        s = starts[start_idx]
        covered_expr.append(regular[s])
        covered_expr.append(contract[s])
    solver.Add(sum(covered_expr) >= period["required_nurses"])

regular_shift_cost = inp["nurse_types"]["regular"]["hourly_pay"] * inp["shift_length_hours"]
contract_shift_cost = inp["nurse_types"]["contract"]["hourly_pay"] * inp["shift_length_hours"]

solver.Minimize(
    sum(regular[s] * regular_shift_cost + contract[s] * contract_shift_cost for s in starts)
)

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}
status_str = status_map.get(status, str(status))

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    regular_nurses_by_start_time = {s: int(round(regular[s].solution_value())) for s in starts}
    contract_nurses_by_start_time = {s: int(round(contract[s].solution_value())) for s in starts}
    total_regular_nurses = sum(regular_nurses_by_start_time.values())
    total_contract_nurses = sum(contract_nurses_by_start_time.values())
    objective_value = int(round(solver.Objective().Value()))
    output = {
        "regular_nurses_by_start_time": regular_nurses_by_start_time,
        "contract_nurses_by_start_time": contract_nurses_by_start_time,
        "total_regular_nurses": total_regular_nurses,
        "total_contract_nurses": total_contract_nurses,
        "hire_contract_nurses": total_contract_nurses > 0,
        "total_cost_yuan": objective_value
    }
else:
    objective_value = None
    output = {
        "regular_nurses_by_start_time": {s: 0 for s in starts},
        "contract_nurses_by_start_time": {s: 0 for s in starts},
        "total_regular_nurses": 0,
        "total_contract_nurses": 0,
        "hire_contract_nurses": False,
        "total_cost_yuan": None
    }

print(json.dumps({
    "status": status_str,
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))