import cv2
import pandas as pd
import argparse
import os

def process_video_events(video_path, evtest_path, output_csv, start_frame=13, time_offset_ms=0.0):
    """
    Matches video frames with joystick events and exports to CSV.

    Args:
        video_path: Path to the mp4 file.
        evtest_path: Path to the .evtest log.
        output_csv: Where to save the result.
        start_frame: The frame index to start processing from.
        time_offset_ms: Manual sync adjustment in milliseconds. 
                        Positive moves the data 'later', negative 'earlier'.
    """
    if not os.path.exists(video_path) or not os.path.exists(evtest_path):
        print(f"Error: Files not found. Check paths: {video_path}, {evtest_path}")
        return

    # 1. Load Evtest Data
    with open(evtest_path, 'r') as f:
        lines = f.readlines()

    # Constants for evtest structure
    HEADER_OFFSET = 77
    # Convert the offset from ms to seconds
    offset_sec = time_offset_ms / 1000.0

    # Get initial absolute timestamp from the log and apply our manual offset
    base_timestamp = float(lines[HEADER_OFFSET].split(' ')[2][:-1]) + offset_sec
    events = lines[HEADER_OFFSET:]

    # Initial state (defaults to center/zero)
    current_state = {'throttle': 0, 'pitch': 0, 'roll': 0, 'yaw': 0}

    # 2. Initialize Video Capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    dataset = []

    print(f"Processing with offset: {time_offset_ms}ms...")

    # 3. Synchronized Processing Loop
    for frame_idx in range(start_frame, total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate 'Sync Time': Where we are in the video relative to the log
        # Frame time = (current frame / fps) + log start time
        current_frame_sync_time = (frame_idx / fps) + base_timestamp

        # Consume all event logs that occurred before this frame's timestamp
        while events:
            # Parse the timestamp of the next event in the queue
            try:
                event_time = float(events[0].split(' ')[2][:-1])
            except (IndexError, ValueError):
                break

            if current_frame_sync_time > event_time:
                event = events.pop(0)

                # Update the specific axis that moved
                if 'ABS_Z' in event:
                    current_state['throttle'] = int(event.strip().split(' ')[-1])
                elif 'ABS_Y' in event:
                    current_state['pitch'] = int(event.strip().split(' ')[-1])
                elif 'ABS_RX' in event:
                    current_state['roll'] = int(event.strip().split(' ')[-1])
                elif 'ABS_X' in event:
                    current_state['yaw'] = int(event.strip().split(' ')[-1])
            else:
                # Event is in the future (relative to this frame), stop consuming
                break

        # Append the state of the sticks at this exact frame moment
        dataset.append({
            'frame': frame_idx,
            'timestamp': current_frame_sync_time,
            **current_state
        })

        if frame_idx % 500 == 0:
            print(f"Frame {frame_idx}/{total_frames} matched.")

    # 4. Save to Disk
    df = pd.DataFrame(dataset).set_index('frame')
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv)

    cap.release()
    print(f"Successfully exported data to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--evtest", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--offset", type=float, default=0.0, help="Offset in milliseconds (e.g. 150.0 or -50.0)")
    parser.add_argument("--start", type=int, default=13)

    args = parser.parse_args()
    process_video_events(args.video, args.evtest, args.out, args.start, args.offset)
