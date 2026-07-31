import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
})

df = pd.read_csv("crash_metric.csv")

yaw = df["yaw"].values
samples = range(len(yaw))

H_YAW = 35



fig, ax = plt.subplots(figsize=(6.5, 3.5))
ymax = max(max(yaw), H_YAW)
ax.set_ylim(0, ymax * 1.75)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

for spine in ax.spines.values():
    spine.set_color("black")
    spine.set_linewidth(1.0)

# =========================
# SEÑAL
# =========================

ax.plot(
    samples,
    yaw,
    linewidth=1.5,
    label=r"$|\Delta \omega_z|$"
)

# =========================
# UMBRAL
# =========================

ax.axhline(
    H_YAW,
    linestyle="--",
    linewidth=1.2,
    label=r"Umbral $H_{\mathrm{yaw}}$"
)

# =========================
# DETECCIÓN
# =========================

collision_idx = None
for i, a in enumerate(yaw):
    if a > H_YAW:
        collision_idx = i
        break

if collision_idx is not None:
    ax.scatter(
        collision_idx,
        yaw[collision_idx],
        s=40,
        zorder=3,
        label="Colisión"
    )

# =========================
# ETIQUETA
# =========================

ax.set_xlabel("Muestra")
ax.set_ylabel(r"Variación de velocidad angular (rad/s)")

ax.grid(True, alpha=0.3)

# ✔ SOLO UNA VEZ Y AL FINAL
ax.legend(loc="upper left", frameon=True)

plt.tight_layout()
plt.savefig("crash_yaw.pdf", bbox_inches="tight")
plt.show()