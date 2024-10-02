import cv2
import numpy as np
import pandas as pd
#load evtest file
with open('StableFlight_10FPS_ShortCircuit_SimpleQuality_2024-8-1.evtest') as f:
    lines = f.readlines()
    throttle = int(lines[44].strip().split(' ')[-1])
    pitch = int(lines[38].strip().split(' ')[-1])
    roll = int(lines[50].strip().split(' ')[-1])
    yaw = int(lines[32].strip().split(' ')[-1])
start_time = float(lines[77].split(' ')[2][:-1])
events = lines[77:]
# Load the video
start = 13
frames = []
cap = cv2.VideoCapture('StableFlight_10FPS_ShortCircuit_SimpleQuality_2024-8-1.webm')
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for f in range(start, total):
    #display progress bar based on total frames
    if f % 100 == 0:
        print(f, total)    
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap.read()
    time = f / fps + start_time
    while events and time > float(events[0].split(' ')[2][:-1]):
        event = events.pop(0)
        if 'ABS_Z' in event:
            throttle = int(event.strip().split(' ')[-1])
        elif 'ABS_Y' in event:
            pitch = int(event.strip().split(' ')[-1])
        elif 'ABS_RX' in event:
            roll = int(event.strip().split(' ')[-1])
        elif 'ABS_X' in event:
            yaw = int(event.strip().split(' ')[-1])

    # store the frame and the event
    frames.append({'frame': f, 'throttle': throttle, 'pitch': pitch, 'roll': roll, 'yaw': yaw})

#store the frames in a dataframe, taking the frame as index
dataFrame = pd.DataFrame(frames).set_index('frame')

dataFrame.to_csv('StableFlight_10FPS_ShortCircuit_SimpleQuality_2024-8-1.csv')

    