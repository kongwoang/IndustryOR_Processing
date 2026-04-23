import json
from ortools.linear_solver import pywraplp

inp = {
    "raw_pipe_length_mm": 1850,
    "raw_pipe_cost_in_raw_pipe_value_units": 1.0,
    "max_patterns_used": 4,
    "max_pieces_per_pattern": 5,
    "max_leftover_mm": 100,
    "rank_extra_cost_once_in_raw_pipe_value_units": [0.1, 0.2, 0.3, 0.4],
    "items": [
        {"length_mm": 290, "demand": 15},
        {"length_mm": 315, "demand": 28},
        {"length_mm": 350, "demand": 21},
        {"length_mm": 455, "demand": 30}
    ]
}

raw_len = inp["raw_pipe_length_mm"]
lengths = [item["length_mm"] for item in inp["items"]]
demands = [item["demand"] for item in inp["items"]]
max_patterns_used = inp["max_patterns_used"]
max_pieces_per_pattern = inp["max_pieces_per_pattern"]
max_leftover_mm = inp["max_leftover_mm"]
rank_penalties = inp["rank_extra_cost_once_in_raw_pipe_value_units"]

patterns = []

def generate_patterns(i, remaining_pieces, counts):
    if i == len(lengths):
        total_pieces = sum(counts)
        if 1 <= total_pieces <= max_pieces_per_pattern:
            used_length = sum(c * l for c, l in zip(counts, lengths))
            leftover = raw_len - used_length
            if 0 <= leftover <= max_leftover_mm:
                patterns.append({
                    "pattern_id": len(patterns) + 1,
                    "pieces": counts[:],
                    "used_length_mm": used_length,
                    "leftover_mm": leftover
                })
        return

    for c in range(remaining_pieces + 1):
        counts.append(c)
        generate_patterns(i + 1, remaining_pieces - c, counts)
        counts.pop()

generate_patterns(0, max_pieces_per_pattern, [])

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    solver = pywraplp.Solver.CreateSolver("CBC")
if solver is None:
    raise RuntimeError("No suitable MIP solver available in OR-Tools.")

P = range(len(patterns))
R = range(1, max_patterns_used + 1)

# Upper bound on number of raw pipes
# Safe bound: use enough pipes to make total demanded pieces, since each pipe yields at least 1 piece
M = sum(demands)

# x[p] = number of times pattern p is used
x = {p: solver.IntVar(0, M, f"x_{p}") for p in P}

# z[p,r] = 1 if pattern p is assigned rank r (r=1 means most used among selected patterns)
z = {(p, r): solver.BoolVar(f"z_{p}_{r}") for p in P for r in R}

# Each pattern can get at most one rank
for p in P:
    solver.Add(sum(z[p, r] for r in R) <= 1)
    # if pattern used, it must receive a rank
    solver.Add(x[p] <= M * sum(z[p, r] for r in R))
    # if it gets a rank, it must be used at least once
    solver.Add(x[p] >= sum(z[p, r] for r in R))

# Each rank assigned to at most one pattern
for r in R:
    solver.Add(sum(z[p, r] for p in P) <= 1)

# Frequency ordering:
# if p has better rank than q, then x[p] >= x[q]
for p in P:
    for q in P:
        if p == q:
            continue
        for r in R:
            for s in R:
                if r < s:
                    solver.Add(x[p] >= x[q] - M * (2 - z[p, r] - z[q, s]))

# Meet demand (allow overproduction if necessary)
for i in range(len(lengths)):
    solver.Add(sum(patterns[p]["pieces"][i] * x[p] for p in P) >= demands[i])

# Objective:
# raw pipe cost per pipe + one-time extra cost for each selected pattern by rank
objective = solver.Objective()
for p in P:
    objective.SetCoefficient(x[p], inp["raw_pipe_cost_in_raw_pipe_value_units"])
    for r in R:
        objective.SetCoefficient(z[p, r], rank_penalties[r - 1])
objective.SetMinimization()

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

output = None
objective_value = None

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    produced = {str(lengths[i]): 0 for i in range(len(lengths))}
    total_raw_pipes = 0
    total_leftover_mm = 0
    selected_patterns = []

    rank_of_pattern = {}
    for p in P:
        for r in R:
            if z[p, r].solution_value() > 0.5:
                rank_of_pattern[p] = r

    for p in P:
        times_used = int(round(x[p].solution_value()))
        if times_used <= 0:
            continue

        rank = rank_of_pattern[p]
        total_raw_pipes += times_used
        total_leftover_mm += times_used * patterns[p]["leftover_mm"]

        for i in range(len(lengths)):
            produced[str(lengths[i])] += times_used * patterns[p]["pieces"][i]

        selected_patterns.append({
            "rank": rank,
            "pattern_id": patterns[p]["pattern_id"],
            "times_used": times_used,
            "pieces_per_raw_pipe": {
                str(lengths[i]): patterns[p]["pieces"][i] for i in range(len(lengths))
            },
            "leftover_mm_per_raw_pipe": patterns[p]["leftover_mm"],
            "raw_pipe_cost_total_in_raw_pipe_value_units": times_used * inp["raw_pipe_cost_in_raw_pipe_value_units"],
            "rank_extra_cost_once_in_raw_pipe_value_units": rank_penalties[rank - 1]
        })

    selected_patterns.sort(key=lambda t: t["rank"])

    overproduction = {
        str(lengths[i]): produced[str(lengths[i])] - demands[i]
        for i in range(len(lengths))
    }

    output = {
        "selected_patterns": selected_patterns,
        "total_raw_pipes": total_raw_pipes,
        "produced_quantities": produced,
        "overproduction": overproduction,
        "total_leftover_mm": total_leftover_mm
    }
    objective_value = solver.Objective().Value()

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result, ensure_ascii=False))