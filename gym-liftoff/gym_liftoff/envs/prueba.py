import subprocess
import re
import mss
import numpy as np
import cv2
import os


def find_liftoff_window(executable_name="Liftoff.x86_64", window_name="LiftOff"):
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


def get_window_position(window_id):
    """Obtiene la posición absoluta de la ventana."""
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




def capture_monitor(monitor):
    """Captura el monitor indicado."""
    with mss.mss() as sct:
        frame = np.array(sct.grab(monitor))[:, :, :3]
    return frame


def save_frame(frame):
    """Guarda la imagen en el mismo directorio del script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "liftoff_capture.png")

    print(frame.shape)
    print(frame.dtype)

    frame_bgr = frame#cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, frame_bgr)
    print("Imagen guardada en:", path)


# ---------------- MAIN ----------------

window_name = "Liftoff"

window_id = find_liftoff_window()

if window_id is None:
    raise RuntimeError("No se encontró la ventana del juego.")

monitor = get_window_position(window_id)

if monitor is None:
    raise RuntimeError("No se pudo detectar el monitor del juego.")

frame = capture_monitor(monitor)

save_frame(frame)

print("Monitor detectado:", monitor)
print("Resolución capturada:", frame.shape)