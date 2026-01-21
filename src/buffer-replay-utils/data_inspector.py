import pandas as pd
import matplotlib.pyplot as plt
import argparse

def inspect_flight_data(csv_path):
    """Generates a visualization of joystick inputs over time."""
    df = pd.read_csv(csv_path)

    # We plot the 4 main axes
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(df['time'], df['throttle'], color='red', label='Throttle')
    plt.ylabel('Value')
    plt.legend(loc='upper right')
    plt.title('Flight Data Synchronization Check')

    plt.subplot(4, 1, 2)
    plt.plot(df['time'], df['pitch'], color='blue', label='Pitch')
    plt.ylabel('Value')
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 3)
    plt.plot(df['time'], df['roll'], color='green', label='Roll')
    plt.ylabel('Value')
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 4)
    plt.plot(df['time'], df['yaw'], color='purple', label='Yaw')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Value')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to synchronized CSV")
    args = parser.parse_args()

    inspect_flight_data(args.csv)
