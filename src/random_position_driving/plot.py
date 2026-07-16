import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_curve_evaluation.csv")

x = df["steps"]

plt.figure(figsize=(6.5,3.5))

plt.plot(x, df["mean_reward"], linewidth=2, label="Mean reward")

plt.fill_between(
    x,
    df["mean_reward"] - df["std_reward"],
    df["mean_reward"] + df["std_reward"],
    alpha=0.2
)

plt.xlabel("Training steps")
plt.ylabel("Episode reward")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_curve_eval.pdf")
plt.show()
