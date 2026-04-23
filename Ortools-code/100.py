import json
from ortools.linear_solver import pywraplp

inp = {
    "nodes": ["A", "B", "C", "D", "E"],
    "bandwidth_matrix": [
        [0, 90, 85, 0, 65],
        [95, 0, 70, 65, 34],
        [60, 0, 0, 88, 80],
        [67, 30, 25, 0, 84],
        [0, 51, 0, 56, 0]
    ],
    "source": "A",
    "target": "E",
    "must_pass_through": ["C"]
}

def clean_num(x):
    if x is None:
        return None
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    return x

nodes = inp["nodes"]
matrix = inp["bandwidth_matrix"]
source = inp["source"]
target = inp["target"]
must_pass = set(inp["must_pass_through"])

n = len(nodes)
idx = {node: i for i, node in enumerate(nodes)}

solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("SCIP solver is not available.")

arcs = []
cap = {}
for i in range(n):
    for j in range(n):
        if i != j and matrix[i][j] > 0:
            arcs.append((i, j))
            cap[(i, j)] = matrix[i][j]

x = {(i, j): solver.IntVar(0, 1, f"x_{i}_{j}") for (i, j) in arcs}
z = solver.NumVar(0, solver.infinity(), "z")
u = {i: solver.NumVar(0, n - 1, f"u_{i}") for i in range(n)}

max_cap = max(cap.values())

out_arcs = {i: [] for i in range(n)}
in_arcs = {i: [] for i in range(n)}
for i, j in arcs:
    out_arcs[i].append((i, j))
    in_arcs[j].append((i, j))

s = idx[source]
t = idx[target]

# Source and target constraints
solver.Add(solver.Sum(x[a] for a in out_arcs[s]) == 1)
solver.Add(solver.Sum(x[a] for a in in_arcs[s]) == 0)

solver.Add(solver.Sum(x[a] for a in in_arcs[t]) == 1)
solver.Add(solver.Sum(x[a] for a in out_arcs[t]) == 0)

# Flow conservation and degree limits
for i in range(n):
    if i in (s, t):
        continue
    in_sum = solver.Sum(x[a] for a in in_arcs[i])
    out_sum = solver.Sum(x[a] for a in out_arcs[i])
    solver.Add(in_sum == out_sum)
    solver.Add(in_sum <= 1)
    solver.Add(out_sum <= 1)

# Must-pass-through nodes
for node in must_pass:
    i = idx[node]
    solver.Add(solver.Sum(x[a] for a in in_arcs[i]) == 1)
    solver.Add(solver.Sum(x[a] for a in out_arcs[i]) == 1)

# Maximize bottleneck bandwidth
for (i, j) in arcs:
    solver.Add(z <= cap[(i, j)] * x[(i, j)] + max_cap * (1 - x[(i, j)]))

# Subtour elimination (MTZ)
solver.Add(u[s] == 0)
for i, j in arcs:
    solver.Add(u[j] >= u[i] + 1 - n * (1 - x[(i, j)]))

solver.Maximize(z)

status = solver.Solve()

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

output = {
    "path": [],
    "max_bandwidth": None
}

objective_value = None

if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
    succ = {}
    for (i, j), var in x.items():
        if var.solution_value() > 0.5:
            succ[i] = j

    path = [source]
    cur = s
    visited = {s}
    while cur != t and cur in succ:
        cur = succ[cur]
        if cur in visited:
            break
        visited.add(cur)
        path.append(nodes[cur])

    output = {
        "path": path,
        "max_bandwidth": clean_num(z.solution_value())
    }
    objective_value = clean_num(solver.Objective().Value())

result = {
    "status": status_map.get(status, str(status)),
    "objective_value": objective_value,
    "example_output": output
}

print(json.dumps(result))