import pandas as pd
import matplotlib.pyplot as plt

# Estilo similar a LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

df = pd.read_csv("training_position.csv")

x = df["steps"]
y = df["mean_reward"]
std = df["std_reward"]

fig, ax = plt.subplots(figsize=(6, 3.5))

ax.plot(
    x,
    y,
    linewidth=1.5,
    color="tab:orange",
    label="Recompensa media"
)

ax.fill_between(
    x,
    y - std,
    y + std,
    alpha=0.25,
    color = "gray",
    label=r"$\pm \sigma$"
)

ax.set_xlabel("Episodios de entrenamiento")
ax.set_ylabel("Recompensa de evaluación")

ax.grid(True, alpha=0.3)
ax.legend(loc="lower left", frameon=True)
plt.tight_layout()
plt.savefig(
    "random_position_training.pdf",
    bbox_inches="tight"
)

plt.show()