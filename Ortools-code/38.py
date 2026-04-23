import json
from ortools.sat.python import cp_model

inp = {
    "products": [1, 2, 3],
    "machines": [1, 2, 3],
    "processing_times": [
        [2, 3, 1],
        [4, 2, 3],
        [3, 5, 2]
    ]
}

products = inp["products"]
machines = inp["machines"]
processing_times = inp["processing_times"]

n_products = len(products)
n_machines = len(machines)
horizon = sum(sum(row) for row in processing_times)

model = cp_model.CpModel()

# x[pos, j] = 1 if product j is placed at sequence position pos
x = {}
for pos in range(n_products):
    for j in range(n_products):
        x[pos, j] = model.NewBoolVar(f"x_{pos}_{j}")

# Each position gets exactly one product
for pos in range(n_products):
    model.Add(sum(x[pos, j] for j in range(n_products)) == 1)

# Each product appears exactly once
for j in range(n_products):
    model.Add(sum(x[pos, j] for pos in range(n_products)) == 1)

zero = model.NewConstant(0)

start = {}
end = {}
for pos in range(n_products):
    for m in range(n_machines):
        start[pos, m] = model.NewIntVar(0, horizon, f"start_{pos}_{m}")
        end[pos, m] = model.NewIntVar(0, horizon, f"end_{pos}_{m}")

        prev_same_machine = end[pos - 1, m] if pos > 0 else zero
        prev_same_product = end[pos, m - 1] if m > 0 else zero

        model.AddMaxEquality(start[pos, m], [prev_same_machine, prev_same_product])

        duration_expr = sum(processing_times[j][m] * x[pos, j] for j in range(n_products))
        model.Add(end[pos, m] == start[pos, m] + duration_expr)

makespan = end[n_products - 1, n_machines - 1]
model.Minimize(makespan)

solver = cp_model.CpSolver()
status = solver.Solve(model)

status_name = solver.StatusName(status)
objective_value = None
output = {
    "product_order": [],
    "makespan": None,
    "schedule": []
}

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    objective_value = int(solver.Value(makespan))
    product_order = []
    schedule = []

    for pos in range(n_products):
        chosen_j = next(j for j in range(n_products) if solver.Value(x[pos, j]) == 1)
        product_id = products[chosen_j]
        product_order.append(product_id)

        machine_times = []
        for m in range(n_machines):
            machine_times.append({
                "machine": machines[m],
                "start": int(solver.Value(start[pos, m])),
                "end": int(solver.Value(end[pos, m]))
            })

        schedule.append({
            "position": pos + 1,
            "product": product_id,
            "machine_times": machine_times
        })

    output = {
        "product_order": product_order,
        "makespan": objective_value,
        "schedule": schedule
    }

print(json.dumps({
    "status": status_name,
    "objective_value": objective_value,
    "example_output": output
}, separators=(",", ":")))