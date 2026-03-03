import numpy as np
from pathlib import Path
import json

current_path = Path(__file__).resolve().parent
try:
    with open(f"{current_path}/delta_data/data.json", "r") as f:
        deltas = json.load(f)
    max_deltas = np.array([deltas["throttle"], deltas["yaw"], deltas["roll"], deltas["pitch"]])
except FileNotFoundError:
    raise FileNotFoundError


def stability_reward(delta_action):
    return - np.sum(np.where(delta_action > max_deltas, (delta_action - max_deltas)**2, 0))/4