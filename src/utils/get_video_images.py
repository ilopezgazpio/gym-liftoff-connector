import lmdb
import cv2
import os
import numpy as np

def get_images(base_dir, lmdb_path, resize = 256):
    env = lmdb.open(lmdb_path, map_size=40 * 1024**3)  # 40 GB memory allocation

    frame_global_idx = 0

    for folder_name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for file_name in sorted(os.listdir(folder_path)):
            if not file_name.endswith(".mp4"):
                continue

            video_path = os.path.join(folder_path, file_name)
            print(f"Processing: {video_path}")
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0

            with env.begin(write=True) as txn:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Redimensionar frame para entrenamiento
                    frame = cv2.resize(frame, (resize, resize))
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Codificar a PNG para guardar en LMDB
                    _, buffer = cv2.imencode(".png", frame_rgb)
                    # Crear clave única
                    key = f"{frame_global_idx:08d}".encode()
                    # Guardar en LMDB
                    txn.put(key, buffer.tobytes())

                    frame_idx += 1
                    frame_global_idx += 1

                    if frame_idx % 10000 == 0:
                        print(f"{frame_idx} frames processed")

            cap.release()

    env.close()

if __name__ == "__main__":
    base_dir = "/home/sergio/Documentos/tfg/gym-liftoff-connector/data/training_data/videos"
    lmdb_path = "/home/sergio/Documentos/tfg/gym-liftoff-connector/data/training_data/video_images.lmdb"
    get_images(base_dir, lmdb_path)