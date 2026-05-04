import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
MASS = 0.675    # ajusta a tu dron
GRAVITY = 9.81
I_zz = 0.0005
# =========================
# LOAD DATA
# =========================
df = pd.read_csv("liftoff_identification_data.csv")

RPM_TO_RAD = 2 * np.pi / 60

df["m1"] = df["m1"] * RPM_TO_RAD
df["m2"] = df["m2"] * RPM_TO_RAD
df["m3"] = df["m3"] * RPM_TO_RAD
df["m4"] = df["m4"] * RPM_TO_RAD

# =========================
# DERIVADAS
# =========================
df["timestamp"] = df["timestamp"].str.strip("[]")
df["timestamp"] = df["timestamp"].astype(float)
df["dt"] = df["timestamp"].diff()

# aceleración vertical (Unity → Y)
df["ay"] = df["vy"].diff() / df["dt"]

# aceleración angular yaw
wy = df["wz"]

df["wz_dot"] = wy / df["dt"]

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
plt.scatter(X_t, y_t, s=5)

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

A = np.vstack([X_t, np.ones(len(X_t))]).T
kt, b = np.linalg.lstsq(A, y_t, rcond=None)[0]

print("===== RESULTADOS Mediante Minimos Cuadrados=====")
print("k_t =", kt)
print("bias =", b)

y_pred = kt * X_t + b
error = y_t - y_pred

ss_res = np.sum(error**2)
ss_tot = np.sum((y_t - np.mean(y_t))**2)

r2 = 1 - ss_res / ss_tot
print("R2 =", r2)

plt.scatter(X_t, y_t, s=5, label="data")
plt.plot(X_t, y_pred, color="red", label="fit")
plt.legend()
plt.show()