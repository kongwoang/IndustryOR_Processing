import json
from ortools.linear_solver import pywraplp

inp = {
    "students": [
        {
            "id": 1,
            "type": "undergraduate",
            "wage": 10.0,
            "availability_hours": {
                "Monday": 6,
                "Tuesday": 0,
                "Wednesday": 6,
                "Thursday": 0,
                "Friday": 7
            },
            "min_weekly_hours": 8,
            "max_weekly_shifts": 2
        },
        {
            "id": 2,
            "type": "undergraduate",
            "wage": 10.0,
            "availability_hours": {
                "Monday": 0,
                "Tuesday": 8,
                "Wednesday": 9,
                "Thursday": 6,
                "Friday": 0
            },
            "min_weekly_hours": 8,
            "max_weekly_shifts": 2
        },
        {
            "id": 3,
            "type": "undergraduate",
            "wage": 9.9,
            "availability_hours": {
                "Monday": 4,
                "Tuesday": 8,
                "Wednesday": 3,
                "Thursday": 0,
                "Friday": 5
            },
            "min_weekly_hours": 8,
            "max_weekly_shifts": 2
        },
        {
            "id": 4,
            "type": "undergraduate",
            "wage": 9.8,
            "availability_hours": {
                "Monday": 5,
                "Tuesday": 5,
                "Wednesday": 6,
                "Thursday": 0,
                "Friday": 4
            },
            "min_weekly_hours": 8,
            "max_weekly_shifts": 2
        },
        {
            "id": 5,
            "type": "graduate",
            "wage": 10.8,
            "availability_hours": {
                "Monday": 3,
                "Tuesday": 0,
                "Wednesday": 5,
                "Thursday": 8,
                "Friday": 0
            },
            "min_weekly_hours": 7,
            "max_weekly_shifts": 2
        },
        {
            "id": 6,
            "type": "graduate",
            "wage": 11.3,
            "availability_hours": {
                "Monday": 0,
                "Tuesday": 6,
                "Wednesday": 0,
                "Thursday": 6,
                "Friday": 5
            },
            "min_weekly_hours": 7,
            "max_weekly_shifts": 2
        }
    ],
    "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "daily_required_hours": {
        "Monday": 14,
        "Tuesday": 14,
        "Wednesday": 14,
        "Thursday": 14,
        "Friday": 14
    },
    "max_students_per_day": 3
}

days = inp["days"]
students = inp["students"]

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
if solver is None:
    raise RuntimeError("No suitable MIP solver is available in OR-Tools.")

x = {}
y = {}

for s in students:
    sid = s["id"]
    max_shifts = s["max_weekly_shifts"]
    min_hours = s["min_weekly_hours"]
    for d in days:
        a = s["availability_hours"][d]
        x[(sid, d)] = solver.IntVar(0, a, f"x_{sid}_{d}")
        y[(sid, d)] = solver.BoolVar(f"y_{sid}_{d}")
        solver.Add(x[(sid, d)] <= a * y[(sid, d)])
        solver.Add(x[(sid, d)] >= y[(sid, d)])
    solver.Add(solver.Sum(x[(sid, d)] for d in days) >= min_hours)
    solver.Add(solver.Sum(y[(sid, d)] for d in days) <= max_shifts)

for d in days:
    solver.Add(solver.Sum(x[(s["id"], d)] for s in students) == inp["daily_required_hours"][d])
    solver.Add(solver.Sum(y[(s["id"], d)] for s in students) <= inp["max_students_per_day"])

solver.Minimize(
    solver.Sum(s["wage"] * x[(s["id"], d)] for s in students for d in days)
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

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    assigned_hours = {}
    selected_shifts = {}
    total_hours_per_student = {}

    for s in students:
        sid = str(s["id"])
        assigned_hours[sid] = {}
        selected_shifts[sid] = []
        total_hours = 0
        for d in days:
            val = int(round(x[(s["id"], d)].solution_value()))
            assigned_hours[sid][d] = val
            if val > 0:
                selected_shifts[sid].append(d)
            total_hours += val
        total_hours_per_student[sid] = total_hours

    students_scheduled_per_day = {}
    for d in days:
        students_scheduled_per_day[d] = sum(
            1 for s in students if int(round(x[(s["id"], d)].solution_value())) > 0
        )

    output = {
        "assigned_hours": assigned_hours,
        "selected_shifts": selected_shifts,
        "total_hours_per_student": total_hours_per_student,
        "students_scheduled_per_day": students_scheduled_per_day
    }

    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": solver.Objective().Value(),
        "example_output": output
    }
else:
    output = {
        "assigned_hours": {},
        "selected_shifts": {},
        "total_hours_per_student": {},
        "students_scheduled_per_day": {}
    }
    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": None,
        "example_output": output
    }

print(json.dumps(result, ensure_ascii=False))