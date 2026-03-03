import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_deltas_histogram_discrete(csv_path, plot = False, delete_zeros = True):
    """
    Load a CSV file with columns ['frame', 'throttle', 'Pitch', 'Yaw', 'roll'],
    calculates the delta between consecutive frames for each column
    Then it counts the delta values and plots them on a histogram.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values("frame").reset_index(drop=True)

    columns = ["throttle", "pitch", "yaw", "roll"]

    delta_df = pd.DataFrame()
    for col in columns:
        delta = (df[col].shift(-1) - df[col]).abs()
        if delete_zeros:
            delta[delta == 0] = np.nan
        delta_df[col] = delta


    #delta_df = delta_df.iloc[:-1]
    delta_df = delta_df.dropna().astype(int)

    if plot:
        for col in delta_df.columns:
            nonzero_deltas = delta_df[col][delta_df[col] != 0].astype(int)
            vals, counts = np.unique(nonzero_deltas, return_counts=True)
            plt.figure()
            plt.bar(vals, counts)
            plt.title(f"Histograma de {col}")
            plt.xlabel("Delta")
            plt.ylabel("Frecuencia")
            plt.show()

    return delta_df

def conf_interval_95_abs_deltas(delta_df, save_path = None):
    """
    Calculates the 97.5% confidence interval for the column of a DataFrame
    """
    columns = delta_df.columns
    ci_values = {}
    for col in columns:
        deltas = delta_df[col].abs()
        ci_upper = np.percentile(deltas, 97.5)
        ci_abs = abs(discrete2continuous(ci_upper) - discrete2continuous(ci_upper*2))
        print(f"{col}: IC 97.5% [{ci_upper}]")
        print(f"{col} absolute and continuous value: IC 97.5% [{ci_abs}]")
        print()
        ci_values[col] = ci_abs
    if save_path:
        with open(save_path, "w") as f:
            json.dump(ci_values, f, indent=4)
    return ci_values

def discrete2continuous(action):
    return action*2/2047 - 1

if __name__ == "__main__":
    df = plot_deltas_histogram_discrete("/home/sergio/Documentos/tfg/gym-liftoff-connector/data/csv_files/demo_flight.csv", plot=True, delete_zeros=True)
    ci_values = conf_interval_95_abs_deltas(df, save_path="/home/sergio/Documentos/tfg/gym-liftoff-connector/gym-liftoff/gym_liftoff/envs/delta_data/data.json")
