import json
from fractions import Fraction
from ortools.linear_solver import pywraplp

inp = {
    "tasks": {
        "task1": {
            "required_effective_hours": 8400,
            "methods": {
                "A": {
                    "team": {
                        "skilled": 1,
                        "laborers": 0
                    },
                    "fixed_cost": 0
                },
                "B": {
                    "team": {
                        "skilled": 1,
                        "laborers": 2
                    },
                    "fixed_cost": 500
                }
            }
        },
        "task2": {
            "required_effective_hours": 10800,
            "methods": {
                "A": {
                    "team": {
                        "skilled": 1,
                        "laborers": 0
                    },
                    "fixed_cost": 0
                },
                "B": {
                    "team": {
                        "skilled": 0,
                        "laborers": 1
                    },
                    "fixed_cost": 0
                }
            }
        },
        "task3": {
            "required_effective_hours": 18000,
            "methods": {
                "A": {
                    "team": {
                        "skilled": 0,
                        "laborers": 5
                    },
                    "fixed_cost": 0
                },
                "B": {
                    "team": {
                        "skilled": 1,
                        "laborers": 3
                    },
                    "fixed_cost": 0
                }
            }
        }
    },
    "weekly_wage": {
        "skilled": 100,
        "laborers": 80
    },
    "effective_hours_per_worker": {
        "skilled": 42,
        "laborers": 36
    },
    "worker_limits": {
        "skilled_max": 400,
        "laborers_max": 800
    },
    "policies": {
        "task1_B_excludes_task3_A": True,
        "task3_B_min_skilled": 20,
        "skilled_at_most_fraction_of_laborers": 0.6
    }
}

solver = pywraplp.Solver.CreateSolver("CBC")
if solver is None:
    solver = pywraplp.Solver.CreateSolver("SCIP")
if solver is None:
    raise RuntimeError("No suitable MIP solver available in OR-Tools.")

# Method selection binaries
x1A = solver.BoolVar("x1A")
x1B = solver.BoolVar("x1B")
x2A = solver.BoolVar("x2A")
x2B = solver.BoolVar("x2B")
x3A = solver.BoolVar("x3A")
x3B = solver.BoolVar("x3B")

# Worker/group count variables
s1A = solver.IntVar(0, inp["worker_limits"]["skilled_max"], "s1A")  # skilled workers on Task 1 via A
g1B = solver.IntVar(0, inp["worker_limits"]["skilled_max"], "g1B")  # groups on Task 1 via B

s2A = solver.IntVar(0, inp["worker_limits"]["skilled_max"], "s2A")  # skilled workers on Task 2 via A
l2B = solver.IntVar(0, inp["worker_limits"]["laborers_max"], "l2B")  # laborers on Task 2 via B

g3A = solver.IntVar(0, inp["worker_limits"]["laborers_max"] // 5, "g3A")  # 5-laborer groups on Task 3 via A
g3B = solver.IntVar(0, min(inp["worker_limits"]["skilled_max"], inp["worker_limits"]["laborers_max"] // 3), "g3B")  # mixed groups on Task 3 via B

# Exactly one method per task
solver.Add(x1A + x1B == 1)
solver.Add(x2A + x2B == 1)
solver.Add(x3A + x3B == 1)

# Method activation and task completion constraints
solver.Add(inp["effective_hours_per_worker"]["skilled"] * s1A >= inp["tasks"]["task1"]["required_effective_hours"] * x1A)
solver.Add(s1A <= inp["worker_limits"]["skilled_max"] * x1A)

solver.Add((inp["effective_hours_per_worker"]["skilled"] + 2 * inp["effective_hours_per_worker"]["laborers"]) * g1B >= inp["tasks"]["task1"]["required_effective_hours"] * x1B)
solver.Add(g1B <= inp["worker_limits"]["skilled_max"] * x1B)

solver.Add(inp["effective_hours_per_worker"]["skilled"] * s2A >= inp["tasks"]["task2"]["required_effective_hours"] * x2A)
solver.Add(s2A <= inp["worker_limits"]["skilled_max"] * x2A)

solver.Add(inp["effective_hours_per_worker"]["laborers"] * l2B >= inp["tasks"]["task2"]["required_effective_hours"] * x2B)
solver.Add(l2B <= inp["worker_limits"]["laborers_max"] * x2B)

solver.Add((5 * inp["effective_hours_per_worker"]["laborers"]) * g3A >= inp["tasks"]["task3"]["required_effective_hours"] * x3A)
solver.Add(g3A <= (inp["worker_limits"]["laborers_max"] // 5) * x3A)

solver.Add((inp["effective_hours_per_worker"]["skilled"] + 3 * inp["effective_hours_per_worker"]["laborers"]) * g3B >= inp["tasks"]["task3"]["required_effective_hours"] * x3B)
solver.Add(g3B <= min(inp["worker_limits"]["skilled_max"], inp["worker_limits"]["laborers_max"] // 3) * x3B)

# Policy constraints
solver.Add(x1B + x3A <= 1)
solver.Add(g3B >= inp["policies"]["task3_B_min_skilled"] * x3B)

total_skilled = solver.IntVar(0, inp["worker_limits"]["skilled_max"], "total_skilled")
total_laborers = solver.IntVar(0, inp["worker_limits"]["laborers_max"], "total_laborers")

solver.Add(total_skilled == s1A + g1B + s2A + g3B)
solver.Add(total_laborers == 2 * g1B + l2B + 5 * g3A + 3 * g3B)

solver.Add(total_skilled <= inp["worker_limits"]["skilled_max"])
solver.Add(total_laborers <= inp["worker_limits"]["laborers_max"])

ratio = Fraction(str(inp["policies"]["skilled_at_most_fraction_of_laborers"]))
solver.Add(ratio.denominator * total_skilled <= ratio.numerator * total_laborers)

# Objective: wages + fixed cost
objective = solver.Objective()
objective.SetMinimization()
objective.SetCoefficient(total_skilled, inp["weekly_wage"]["skilled"])
objective.SetCoefficient(total_laborers, inp["weekly_wage"]["laborers"])
objective.SetCoefficient(x1B, inp["tasks"]["task1"]["methods"]["B"]["fixed_cost"])

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
    def ivar(v):
        return int(round(v.solution_value()))

    chosen_methods = {
        "task1": "A" if x1A.solution_value() > 0.5 else "B",
        "task2": "A" if x2A.solution_value() > 0.5 else "B",
        "task3": "A" if x3A.solution_value() > 0.5 else "B",
    }

    task1_skilled = ivar(s1A) + ivar(g1B)
    task1_laborers = 2 * ivar(g1B)

    task2_skilled = ivar(s2A)
    task2_laborers = ivar(l2B)

    task3_skilled = ivar(g3B)
    task3_laborers = 5 * ivar(g3A) + 3 * ivar(g3B)

    output = {
        "chosen_methods": chosen_methods,
        "hired_workers": {
            "skilled": ivar(total_skilled),
            "laborers": ivar(total_laborers)
        },
        "task_assignments": {
            "task1": {
                "skilled": task1_skilled,
                "laborers": task1_laborers
            },
            "task2": {
                "skilled": task2_skilled,
                "laborers": task2_laborers
            },
            "task3": {
                "skilled": task3_skilled,
                "laborers": task3_laborers
            }
        }
    }

    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": int(round(solver.Objective().Value())),
        "example_output": output
    }
else:
    result = {
        "status": status_map.get(status, str(status)),
        "objective_value": None,
        "example_output": {
            "chosen_methods": {
                "task1": "",
                "task2": "",
                "task3": ""
            },
            "hired_workers": {
                "skilled": 0,
                "laborers": 0
            },
            "task_assignments": {
                "task1": {
                    "skilled": 0,
                    "laborers": 0
                },
                "task2": {
                    "skilled": 0,
                    "laborers": 0
                },
                "task3": {
                    "skilled": 0,
                    "laborers": 0
                }
            }
        }
    }

print(json.dumps(result, ensure_ascii=False))