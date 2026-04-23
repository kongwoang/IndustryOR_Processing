import json
import math
from ortools.linear_solver import pywraplp

inp = {
    "targets": [
        {
            "target_id": 1,
            "distance_km": 450,
            "p_destroy_per_heavy_bomb": 0.03,
            "p_destroy_per_light_bomb": 0.08
        },
        {
            "target_id": 2,
            "distance_km": 480,
            "p_destroy_per_heavy_bomb": 0.1,
            "p_destroy_per_light_bomb": 0.11
        },
        {
            "target_id": 3,
            "distance_km": 540,
            "p_destroy_per_heavy_bomb": 0.05,
            "p_destroy_per_light_bomb": 0.12
        },
        {
            "target_id": 4,
            "distance_km": 600,
            "p_destroy_per_heavy_bomb": 0.05,
            "p_destroy_per_light_bomb": 0.09
        }
    ],
    "resources": {
        "max_heavy_bombs": 28,
        "max_light_bombs": 12,
        "max_fuel_liters": 10000
    },
    "fuel_rules": {
        "km_per_liter_loaded": {
            "heavy": 2,
            "light": 3
        },
        "km_per_liter_empty": 4,
        "takeoff_landing_fuel_per_trip_liters": 100
    }
}

def floor4(x):
    return math.floor(x * 10000.0) / 10000.0

def fuel_per_trip(distance_km, bomb_type, fuel_rules):
    return (
        distance_km / fuel_rules["km_per_liter_loaded"][bomb_type]
        + distance_km / fuel_rules["km_per_liter_empty"]
        + fuel_rules["takeoff_landing_fuel_per_trip_liters"]
    )

def destroy_prob(p_h, p_l, h, l):
    return 1.0 - ((1.0 - p_h) ** h) * ((1.0 - p_l) ** l)

def pair_pmf(q1, q2):
    return (
        (1.0 - q1) * (1.0 - q2),
        q1 * (1.0 - q2) + (1.0 - q1) * q2,
        q1 * q2
    )

def pmf_dominates(pmf_a, pmf_b, eps=1e-12):
    tail_a = 0.0
    tail_b = 0.0
    for k in range(len(pmf_a) - 1, 0, -1):
        tail_a += pmf_a[k]
        tail_b += pmf_b[k]
        if tail_a + eps < tail_b:
            return False
    return True

def prune_pair_states(states):
    states = sorted(
        states,
        key=lambda s: (s["heavy"], s["light"], s["fuel2"], -s["pmf"][2], -s["pmf"][1])
    )
    keep = []
    for cand in states:
        dominated = False
        remove_ids = []
        for i, kept in enumerate(keep):
            kept_better_resources = (
                kept["heavy"] <= cand["heavy"]
                and kept["light"] <= cand["light"]
                and kept["fuel2"] <= cand["fuel2"]
            )
            cand_better_resources = (
                cand["heavy"] <= kept["heavy"]
                and cand["light"] <= kept["light"]
                and cand["fuel2"] <= kept["fuel2"]
            )
            if kept_better_resources and pmf_dominates(kept["pmf"], cand["pmf"]):
                dominated = True
                break
            if cand_better_resources and pmf_dominates(cand["pmf"], kept["pmf"]):
                remove_ids.append(i)
        if not dominated:
            for i in reversed(remove_ids):
                keep.pop(i)
            keep.append(cand)
    return keep

solver = pywraplp.Solver.CreateSolver("GLOP")
if solver is None:
    raise RuntimeError("OR-Tools solver is unavailable.")

targets = inp["targets"]
max_heavy = inp["resources"]["max_heavy_bombs"]
max_light = inp["resources"]["max_light_bombs"]
max_fuel2 = int(round(2 * inp["resources"]["max_fuel_liters"]))

target_options = []
for t in targets:
    fuel_h = fuel_per_trip(t["distance_km"], "heavy", inp["fuel_rules"])
    fuel_l = fuel_per_trip(t["distance_km"], "light", inp["fuel_rules"])
    options = []
    for h in range(max_heavy + 1):
        for l in range(max_light + 1):
            fuel2 = int(round(2 * (h * fuel_h + l * fuel_l)))
            if fuel2 <= max_fuel2:
                options.append({
                    "heavy": h,
                    "light": l,
                    "fuel2": fuel2,
                    "q": destroy_prob(
                        t["p_destroy_per_heavy_bomb"],
                        t["p_destroy_per_light_bomb"],
                        h,
                        l
                    )
                })
    target_options.append(options)

pair_frontiers = []
for a, b in [(0, 1), (2, 3)]:
    states = []
    for opt_a in target_options[a]:
        for opt_b in target_options[b]:
            heavy = opt_a["heavy"] + opt_b["heavy"]
            light = opt_a["light"] + opt_b["light"]
            fuel2 = opt_a["fuel2"] + opt_b["fuel2"]
            if heavy <= max_heavy and light <= max_light and fuel2 <= max_fuel2:
                states.append({
                    "heavy": heavy,
                    "light": light,
                    "fuel2": fuel2,
                    "pmf": pair_pmf(opt_a["q"], opt_b["q"]),
                    "plan": [(opt_a["heavy"], opt_a["light"]), (opt_b["heavy"], opt_b["light"])]
                })
    pair_frontiers.append(prune_pair_states(states))

best_obj = -1.0
best_pair_a = None
best_pair_b = None

for state_a in pair_frontiers[0]:
    rem_heavy = max_heavy - state_a["heavy"]
    rem_light = max_light - state_a["light"]
    rem_fuel2 = max_fuel2 - state_a["fuel2"]
    a0, a1, a2 = state_a["pmf"]

    for state_b in pair_frontiers[1]:
        if (
            state_b["heavy"] <= rem_heavy
            and state_b["light"] <= rem_light
            and state_b["fuel2"] <= rem_fuel2
        ):
            b0, b1, b2 = state_b["pmf"]
            obj = 1.0 - (a0 * b0 + a1 * b0 + a0 * b1)
            if obj > best_obj:
                best_obj = obj
                best_pair_a = state_a
                best_pair_b = state_b

status = "OPTIMAL"
objective_value = None
output = {
    "plan_by_target": [],
    "total_heavy_bombs_used": 0,
    "total_light_bombs_used": 0,
    "total_fuel_used_liters": 0.0,
    "success_probability_at_least_2_destroyed": None
}

if best_pair_a is not None and best_pair_b is not None:
    full_plan = best_pair_a["plan"] + best_pair_b["plan"]

    plan_by_target = []
    total_heavy = 0
    total_light = 0
    total_fuel = 0.0

    for t, (h, l) in zip(targets, full_plan):
        fuel_h = fuel_per_trip(t["distance_km"], "heavy", inp["fuel_rules"])
        fuel_l = fuel_per_trip(t["distance_km"], "light", inp["fuel_rules"])
        fuel_used = h * fuel_h + l * fuel_l
        q = destroy_prob(t["p_destroy_per_heavy_bomb"], t["p_destroy_per_light_bomb"], h, l)

        plan_by_target.append({
            "target_id": t["target_id"],
            "heavy_bombs": h,
            "light_bombs": l,
            "fuel_used_liters": floor4(fuel_used),
            "destroy_probability": floor4(q)
        })

        total_heavy += h
        total_light += l
        total_fuel += fuel_used

    objective_value = floor4(best_obj)
    output = {
        "plan_by_target": plan_by_target,
        "total_heavy_bombs_used": total_heavy,
        "total_light_bombs_used": total_light,
        "total_fuel_used_liters": floor4(total_fuel),
        "success_probability_at_least_2_destroyed": floor4(best_obj)
    }

print(json.dumps({
    "status": status,
    "objective_value": objective_value,
    "example_output": output
}, ensure_ascii=False))