import json
import copy
import numpy as np
import random

flow_ori = json.load(open("/home/wen/tr/LLMTSCS/data/Jinan/3_4/anon_3_4_jinan_real.json", "r"))

all_routes = {}
route2id = {}
r_id = 0
for veh in flow_ori:
    if str(veh["route"]) not in route2id:
        route2id[str(veh["route"])] = r_id
        all_routes[r_id] = {"route": veh["route"], "count": 1}
        r_id += 1
    else:
        route_id = route2id[str(veh["route"])]
        all_routes[route_id]["count"] += 1

probs = {i: all_routes[i]["count"] / len(flow_ori) for i in range(len(all_routes))}

min_num = 0
flow = []

for j in range(8000):  # vehicle number
    time = int(np.random.randint(0, 3599))  # time in seconds
    cumulative_prob = 0
    random_num = random.random()
    for i, p in enumerate(probs):
        cumulative_prob += probs[p]
        if random_num <= cumulative_prob:
            vehicle_id = i + min_num
            break
    veh = copy.deepcopy(flow_ori[0])
    veh["route"] = all_routes[vehicle_id]["route"]
    veh["interval"] = 1.0
    veh["startTime"] = time
    veh["endTime"] = time
    flow.append(veh)


flow.sort(key=lambda x: x["startTime"])

with open("/home/wen/tr/LLMTSCS/data/Jinan/3_4/synthetic_8000_1h.json", "w", encoding="utf-8") as f:
    f.write("[\n")
    for i, veh in enumerate(flow):
        json.dump(veh, f)
        if i != len(flow) - 1:
            f.write(",\n")
    f.write("\n]")
