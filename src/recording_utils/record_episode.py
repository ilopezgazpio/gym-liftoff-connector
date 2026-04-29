from .Record import RecordingTool
import time

record = RecordingTool(zarr_path= None)

print("Empezandoooooo")
time.sleep(5)
record.reset()
while True:
    info = record.get_info()
    time.sleep(1)
    print(info["position"])