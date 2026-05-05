import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
MASS = 0.675
GRAVITY = 9.81
I_zz = 0.005

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("liftoff_identification_data.csv")

RPM_TO_RAD = 2 * np.pi / 60

for m in ["m1", "m2", "m3", "m4"]:
    df[m] = df[m] * RPM_TO_RAD

# =========================
# TIEMPO
# =========================
df["timestamp"] = df["timestamp"].str.strip("[]").astype(float)
df["dt"] = df["timestamp"].diff()

# =========================
# SUAVIZADO (muy importante)
# =========================
df["vy"] = df["vy"].rolling(5).mean()
df["wz"] = df["wz"].rolling(5).mean()

# =========================
# DERIVADAS
# =========================
df["ay"] = df["vy"].diff() / df["dt"]
df["wz_dot"] = df["wz"].diff() / df["dt"]

# momento de yaw
df["Mz"] = I_zz * df["wz_dot"]

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
# FILTRADO GLOBAL
# =========================
df = df[(df["S"] > 1e-3)]
df = df[np.abs(df["ay"]) < 20]
df = df[np.abs(df["wz_dot"]) < 100]

# =========================
# ========= k_t ===========
# =========================
df_t = df[df["phase"] == "thrust"].copy()


X_t = df_t["S"].values
y_t = df_t["T"].values

A_t = np.vstack([X_t, np.ones(len(X_t))]).T
kt, b = np.linalg.lstsq(A_t, y_t, rcond=None)[0]

print("===== k_t =====")
print("k_t =", kt)
print("bias =", b)

# calidad ajuste
y_pred_t = kt * X_t + b
r2_t = 1 - np.sum((y_t - y_pred_t)**2) / np.sum((y_t - np.mean(y_t))**2)
print("R2_t =", r2_t)

plt.figure()
plt.scatter(X_t, y_t, s=5, label="data")
plt.plot(X_t, y_pred_t, color="red", label="fit")
plt.title("Thrust fit")
plt.legend()

# =========================
# ========= k_d ===========
# =========================
df_d = df[df["phase"] == "yaw"].copy()

# quitar muestras sin excitación

X_d = df_d["yaw_term"].values
y_d = df_d["Mz"].values

kd = np.linalg.lstsq(X_d.reshape(-1, 1), y_d, rcond=None)[0][0]

print("\n===== k_d =====")
print("k_d =", kd)

# calidad ajuste
y_pred_d = kd * X_d
r2_d = 1 - np.sum((y_d - y_pred_d)**2) / np.sum((y_d - np.mean(y_d))**2)
print("R2_d =", r2_d)

plt.figure()
plt.scatter(X_d, y_d, s=5, label="data")
plt.plot(X_d, y_pred_d, color="red", label="fit")
plt.title("Yaw fit")
plt.legend()

plt.show()