import subprocess
import re
import mss
import numpy as np
import cv2
import os

def find_liftoff_window(executable_name="LiftOff.x86_64", window_name="LiftOff"):
    try:
        window_ids = subprocess.check_output(["xdotool", "search", "--name", window_name]).split()
    except subprocess.CalledProcessError:
        return None

    for wid in window_ids:
        wid = wid.decode()
        try:
            pid = subprocess.check_output(["xdotool", "getwindowpid", wid]).decode().strip()
            cmd = subprocess.check_output(["ps", "-p", pid, "-o", "cmd="]).decode().strip()
            if executable_name in cmd:
                return wid
        except subprocess.CalledProcessError:
            continue
    return None

def get_monitor(window_name = "LiftOff", executable_name="Liftoff.x86_64"):
    window_id = find_liftoff_window(window_name=window_name, executable_name=executable_name)
    monitor = get_window_position(window_id)
    return monitor

def get_window_position(window_id):
    """Returns the absolute position of the window"""
    geo = subprocess.check_output(["xdotool", "getwindowgeometry", window_id]).decode()

    # Tamaño
    m_geo = re.search(r'Geometry:\s*(\d+)x(\d+)', geo)
    width, height = int(m_geo.group(1)), int(m_geo.group(2))

    # Posición
    m_pos = re.search(r'Position:\s*(\d+),(\d+)', geo)
    if m_pos:
        left, top = int(m_pos.group(1)), int(m_pos.group(2))
    else:
        # fallback por si no aparece
        left, top = 0, 0

    return {"top": top, "left": left, "width": width, "height": height}