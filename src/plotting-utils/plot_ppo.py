import json
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
with open("training_sac_hovering.json", "r") as f:
    data = json.load(f)


rewards = [ep["env_reward_total"] + 97 for ep in data]

import pandas as pd
import json


df = pd.DataFrame({
    "episode": range(len(rewards)),
    "reward": rewards
})

# EMA en vez de rolling simple (más estable visualmente)
df["ema100"] = df["reward"].ewm(span=100, adjust=False).mean()
df["ema3000"] = df["reward"].ewm(span=3000, adjust=False).mean()

# sanity check
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
})
plt.style.use("seaborn-v0_8-white")

fig, ax = plt.subplots(figsize=(9, 4.5))

# =========================
# RAW (ruido)
# =========================
ax.plot(
    df["episode"],
    df["reward"],
    color="gray",
    alpha=0.2,
    linewidth=0.8
)

# =========================
# EMA (señal real)
# =========================5
"""
ax.plot(
    df["episode"],
    df["ema100"],
    linewidth=1.5,
    label="EMA (100)"
)
"""


ax.plot(
    df["episode"],
    df["ema3000"],
    linewidth=2.5,
    color = "tab:orange",
    label="EMA (3000)"
)
# =========================
# FORMATO
# =========================
ax.set_xlabel("Episodio")
ax.set_ylabel("Recompensa acumulada")

ax.grid(True, alpha=0.15, linewidth=0.8)

ax.tick_params(axis='both', which='major', labelsize=10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(["Reward", "EMA (3000)"], loc="upper left", frameon=True)

plt.tight_layout()

plt.savefig("hovering_reward_training.pdf", bbox_inches="tight")
plt.show()