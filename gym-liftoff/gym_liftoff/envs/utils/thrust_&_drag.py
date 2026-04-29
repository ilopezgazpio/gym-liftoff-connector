import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
MASS = 0.675    # ajusta a tu dron
GRAVITY = 9.81

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("liftoff_identification_data.csv")

# =========================
# DERIVADAS
# =========================
df["dt"] = df["timestamp"].diff()

# aceleración vertical (Unity → Y)
df["ay"] = df["vy"].diff() / df["dt"]

# aceleración angular yaw
df["wz_dot"] = df["wz"].diff() / df["dt"]

# eliminar primeras filas
df = df.dropna()

# =========================
# FEATURES MOTORES
# =========================
df["S"] = df["m1"]**2 + df["m2"]**2 + df["m3"]**2 + df["m4"]**2

df["yaw_term"] = (
    df["m1"]**2 - df["m2"]**2 +
    df["m3"]**2 - df["m4"]**2
)

# =========================
# THRUST
# =========================
df["T"] = MASS * (df["ay"] + GRAVITY)

# =========================
# FILTRADO (muy importante)
# =========================
# evitar datos basura (caídas, saturaciones, etc.)
df = df[(df["S"] > 1e-3)]
df = df[np.abs(df["ay"]) < 20]
df = df[np.abs(df["wz_dot"]) < 50]

# =========================
# ESTIMACIÓN kt
# =========================
X_t = df["S"].values
y_t = df["T"].values

kt = np.sum(X_t * y_t) / np.sum(X_t**2)

# =========================
# ESTIMACIÓN kd'
# =========================
X_d = df["yaw_term"].values
y_d = df["wz_dot"].values

kd_prime = np.sum(X_d * y_d) / np.sum(X_d**2)

# =========================
# RESULTADOS
# =========================
print("===== RESULTADOS =====")
print("k_t =", kt)
print("k_d' =", kd_prime)