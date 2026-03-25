import pickle
from pathlib import Path
current_dir = Path(__file__).resolve().parent
lmdb_path = current_dir / "lmdb_episode"
info_path = current_dir / "infos"

# abrir el archivo
for episode in range(100):
    print("Episode:", episode)
    with open(f"{str(info_path)}/{episode}.pkl", "rb") as f:
        episode_data = pickle.load(f)

    for i, step in enumerate(episode_data):
        timestamp = step["timestamp"]
        position = step["position"]  # normalmente un array de 3 elementos
        velocity = step["velocity"]  # array de 3 elementos
        gyro = step["gyro"]  # array de 3 elementos
        input_data = step["input"]
        speed = step["speed"]
        mov = step["movement"]
        inpu = step["input_active"]# array de 4 elementos

        print(f"Step {i}: timestamp={timestamp}, pos={position}, vel={velocity}, gyro={gyro}, input={input_data}, speed={speed}, movement={mov}, input_active={inpu}")

print(f"Número de pasos en el episodio: {len(episode_data)}")