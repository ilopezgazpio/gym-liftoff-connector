# Liftoff Drone Replay Buffer generation: Data Pipeline & Synchronization

This folder provides a pipeline for creating replay buffers for **Liftoff FPV Simulator**.
It specifically solves the problem of synchronizing high-frequency physical stick inputs from a **RadioMaster TX16S** (recorded via `evtest`) with flight video for behavioral cloning.

## 📋 The Pipeline Architecture

To ensure high-precision timing and avoid performance bottlenecks, the process is decoupled into three stages:

1.  **Dataset Generation**: Aligns raw `.evtest` logs with `.mp4` video frames.
2.  **Data Processing**: Normalizes hardware integers (0–2047) into float actions (-1.0 to 1.0).
3.  **Gym Replay**: Feeds actions into the `gym-liftoff` environment via the Steam connector.

---

## 🚀 Getting Started

### 1. Synchronize Video and Logs (`dataset_generator.py`)
This script matches stick movements to video frames. It uses sequential reading to maintain 100% timing accuracy.

**Handling Delays:**
* **`--offset`**: Adjusts for USB latency or recording start-time differences.
    * Use `100.0` to shift data 100ms later.
    * Use `-100.0` to shift data 100ms earlier.
* **`--start`**: Sets the first video frame to process (skips menus).

```bash
python dataset_generator.py \
    --video "recordings/flight_01.mp4" \
    --evtest "logs/flight_01.evtest" \
    --out "data/raw_sync.csv" \
    --offset 150.0 \
    --start 13
```


### 2. Normalize Actions (process_flight_data.py)

Converts raw TX16S integers into normalized floats required by Gymnasium wrappers like LiftoffFloatActions.

```bash
python process_flight_data.py \
    --input "data/raw_sync.csv" \
    --output "data/processed_actions.npy"
```

### 3. Simulator Replay (replay_gym_liftoff.py)

```bash
python replay_gym_liftoff.py --input "data/processed_actions.npy"
```