import pandas as pd
import numpy as np
import argparse

def normalize_tx16s(val, min_val=0, max_val=2047):
    """Converts raw TX16S values to -1.0 to 1.0 range."""
    # Ensure value is within bounds
    val = np.clip(val, min_val, max_val)
    # Center at 0 and scale
    return (val - (max_val / 2)) / (max_val / 2)

def prepare_dataset(input_csv, output_npy):
    """Processes the CSV into a clean numpy array for the Gym Env."""
    df = pd.read_csv(input_csv)

    # Mapping the specific columns we generated in the previous step
    # Order: [throttle, yaw, roll, pitch] as per your play_game logic
    processed_actions = []

    for _, row in df.iterrows():
        action = np.array([
            normalize_tx16s(row['throttle']),
            normalize_tx16s(row['yaw']),
            normalize_tx16s(row['roll']),
            normalize_tx16s(row['pitch'])
        ], dtype=np.float32)
        processed_actions.append(action)

    np.save(output_npy, np.array(processed_actions))
    print(f"Cleaned dataset saved to {output_npy}. Total steps: {len(processed_actions)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='./data/csv_files/demo_flight.csv')
    parser.add_argument("--output", default='./data/processed_actions.npy')
    args = parser.parse_args()
    prepare_dataset(args.input, args.output)
