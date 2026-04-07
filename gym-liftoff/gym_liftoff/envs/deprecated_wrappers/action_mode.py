import numpy as np

def discrete2continuous(action):
    return action*2/2047 - 1

def continuous2discrete(action):
    action = ((action + 1) / 2 * 2047)
    return action.astype(np.uint16)