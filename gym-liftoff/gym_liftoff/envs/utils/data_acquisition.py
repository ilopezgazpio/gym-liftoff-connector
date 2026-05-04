import numpy as np
import pandas as pd
from gym_liftoff.envs.liftoff_env import Liftoff

env = Liftoff(max_episode_time = [1, 0, 0, 0])

# =========================
# CONFIG
# =========================
DT = 0.02  # asumido (ajusta si tu env lo da)

HOVER_THROTTLE = 1024

data = []

# =========================
# UTIL LOG
# =========================
def log(step, phase, action, info):
    data.append({
        "step": step,
        "phase": phase,
        "timestamp": info["timestamp"],

        "x": info["position"][0],
        "y": info["position"][1],
        "z": info["position"][2],

        "vx": info["velocity"][0],
        "vy": info["velocity"][1],
        "vz": info["velocity"][2],

        "wx": info["gyro"][0],
        "wy": info["gyro"][1],
        "wz": info["gyro"][2],

        "m1": info["motorrpm"][0],
        "m2": info["motorrpm"][1],
        "m3": info["motorrpm"][2],
        "m4": info["motorrpm"][3],

        "throttle": action[0],
        "yaw": action[1],
        "roll": action[2],
        "pitch": action[3],
    })

# =========================
# RESET
# =========================
_, info = env.reset()

step = 0

# =========================================================
# FASE 1: THRUST (kt)
# =========================================================
print("Phase 1: thrust identification")

for throttle in range(1200, 2000, 100):

    action = [throttle, 1024, 1024, 1024]

    for _ in range(50):  # estabilización por nivel

        _, _, _, _, info = env.step(action)
        log(step, "thrust", action, info)
        print(info["motorrpm"])
        step += 1

# =========================================================
# FASE 2: YAW (kd)
# =========================================================
print("Phase 2: yaw identification")

action = [HOVER_THROTTLE, 1024, 1024, 1024]

for _ in range(50):
    _, _, _, _, info = env.step(action)
    log(step, "yaw_init", action, info)
    step += 1

for yaw in range(1200, 2000, 100):

    action = [HOVER_THROTTLE, yaw, 1024, 1024]

    for _ in range(20):

        _, _, _, _, info = env.step(action)
        log(step, "yaw", action, info)
        step += 1

# =========================================================
# FASE 3: PERTURBACIONES (robustez)
# =========================================================
print("Phase 3: perturbations")

for _ in range(300):

    action = [
        HOVER_THROTTLE + np.random.randint(-50, 50),
        1024,
        HOVER_THROTTLE + np.random.randint(-20, 20),
        HOVER_THROTTLE + np.random.randint(-20, 20),
    ]

    _, _, _, _, info = env.step(action)
    log(step, "noise", action, info)
    step += 1

# =========================
# SAVE
# =========================
df = pd.DataFrame(data)
df.to_csv("liftoff_identification_data.csv", index=False)

print("Done. Data saved to liftoff_identification_data.csv")