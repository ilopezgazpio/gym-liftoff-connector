import subprocess
import re
import mss
import numpy as np
import cv2
import tkinter as tk
import os

# ---------------- Funciones ----------------

def find_game_window(executable_name, window_name):
    """Devuelve el ID de la ventana principal del juego (solo para X11)."""
    try:
        window_ids = subprocess.check_output(["xdotool", "search", "--name", window_name]).split()
    except subprocess.CalledProcessError:
        return None  # xdotool falla (Wayland)

    for wid in window_ids:
        wid = wid.decode("utf-8")
        try:
            pid = subprocess.check_output(["xdotool", "getwindowpid", wid]).decode().strip()
            cmd = subprocess.check_output(["ps", "-p", pid, "-o", "cmd="]).decode().strip()
            if executable_name in cmd:
                return wid
        except subprocess.CalledProcessError:
            continue
    return None

def get_window_geometry(window_id):
    """Devuelve diccionario {'top','left','width','height'} (solo X11)."""
    geo = subprocess.check_output(["xdotool", "getwindowgeometry", window_id]).decode()

    # Tamaño
    m_geo = re.search(r'Geometry: (\d+)x(\d+)', geo)
    width, height = int(m_geo.group(1)), int(m_geo.group(2))

    # Posición
    m_pos = re.search(r'Absolute upper-left X:\s*(\d+).*Y:\s*(\d+)', geo, re.DOTALL)
    if m_pos:
        left = int(m_pos.group(1))
        top = int(m_pos.group(2))
    else:
        left, top = 0, 0

    return {"top": top, "left": left, "width": width, "height": height}

def get_monitor_geometry(monitor_index=0):
    """Devuelve geometría de un monitor usando Tkinter."""
    root = tk.Tk()
    root.withdraw()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()

    left = monitor_index * width  # monitor horizontal
    top = 0
    return {"top": top, "left": left, "width": width, "height": height}

def capture_window(window_geometry):
    """Captura la ventana o monitor usando MSS."""
    try:
        with mss.mss() as sct:
            frame = np.array(sct.grab(window_geometry))[:, :, :3]
        return frame
    except mss.exception.ScreenShotError as e:
        print("Error capturando pantalla:", e)
        return None

def save_frame(frame, filename="captura.png"):
    """Guarda la imagen en disco."""
    if frame is not None:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, frame_bgr)
        print(f"Imagen guardada en '{filename}' con tamaño {frame.shape}")
    else:
        print("No hay frame para guardar.")

# ---------------- Script principal ----------------

game_executable = "Inscryption.x86_64"
game_window_name = "Inscryption"

# 1️⃣ Intentar usar X11
window_id = find_game_window(game_executable, game_window_name)
if window_id is not None:
    print("Usando X11 para detectar ventana...")
    monitor = get_window_geometry(window_id)
else:
    print("X11 no disponible o Wayland activo, usando monitor completo con Tkinter...")
    monitor = get_monitor_geometry(monitor_index=1)  # segundo monitor

# 2️⃣ Captura y guardado
frame = capture_window(monitor)
save_frame(frame, "inscryption_capture.png")