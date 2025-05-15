import os
import json
import datetime
import numpy as np

START = 106
INTERVAL = 5
REPS = 10
IN_PATH = "results_coco2/sacred/mappo/perturb/minimal-shift/"
OUT_PATH = "results_coco2/shift_eval_summary"
NOTES = "goal shift eval; mappo"
# NOTES = "shelf shift eval; mappo"

results = [[] for _ in range(5)]

for i in range(START, START + REPS * INTERVAL, INTERVAL):
    for j in range(INTERVAL):
        _path = os.path.join(IN_PATH, str(i + j), "metrics.json")
        with open(_path, "r") as in_file:
            d = json.load(in_file)
        results[j].append(d["num_delivers_mean"]["values"][1])


results = np.array(results)
_avg = results.mean(1)
avg = _avg.mean()
std = _avg.std()

date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(os.path.join(OUT_PATH, date_str), exist_ok=True)
with open(os.path.join(OUT_PATH, date_str, "out.txt"), "w") as out_file:
    out_file.write(f"NOTES {NOTES}\n")
    out_file.write(f"START {START}\n")
    out_file.write(f"INTERVAL {INTERVAL}")
    out_file.write(f"REPS {REPS}\n")
    out_file.write(f"IN_PATH {IN_PATH}\n")
    out_file.write(f"num of delivered shelves: {avg} +/- {std}")