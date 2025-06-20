import os
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = "tensorboard_logs/tqc_road"
SCALARS = {
    "rollout/ep_rew_mean": "Episode Reward",
    "rollout/ep_len_mean": "Episode Length"
}
SMOOTHING_WINDOW = 10
SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_scalars(log_path, scalar_key, step_offset):
    ea = event_accumulator.EventAccumulator(log_path)
    try:
        ea.Reload()
        if scalar_key not in ea.scalars.Keys():
            return None, 0
        events = ea.Scalars(scalar_key)
        steps = [e.step + step_offset for e in events]
        values = [e.value for e in events]
        last_step = steps[-1] if steps else 0
        return pd.DataFrame({"step": steps, "value": values}), last_step
    except:
        return None, 0

def merge_all_scalars(log_dir, scalar_key):
    all_data = []
    step_offset = 0

    for root, dirs, files in os.walk(log_dir):
        dirs.sort()  # ensure chronological order
        for d in dirs:
            full_path = os.path.join(root, d)
            df, last_step = load_scalars(full_path, scalar_key, step_offset)
            if df is not None:
                all_data.append(df)
                step_offset = df["step"].max() + 1  # avoid overlaps
    if all_data:
        df_merged = pd.concat(all_data).sort_values("step")
        df_merged["smoothed"] = df_merged["value"].rolling(SMOOTHING_WINDOW, min_periods=1).mean()
        return df_merged
    return None

# Merge both scalars with corrected steps
reward_df = merge_all_scalars(LOG_DIR, "rollout/ep_rew_mean")
length_df = merge_all_scalars(LOG_DIR, "rollout/ep_len_mean")

# Plot with dual y-axis
if reward_df is not None and length_df is not None:
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(reward_df["step"], reward_df["smoothed"], color='tab:blue', label="Episode Reward", linewidth=2)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Episode Reward", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(length_df["step"], length_df["smoothed"], color='tab:orange', label="Episode Length", linewidth=2)
    ax2.set_ylabel("Episode Length", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title("Episode Reward and Length Over Time")
    fig.tight_layout()

    # Save plot
    filepath = os.path.join(SAVE_DIR, "reward_vs_length_road.png")
    plt.savefig(filepath)
    print(f"Saved plot to: {filepath}")

    plt.show()
else:
    print("Could not load one or both metrics.")
