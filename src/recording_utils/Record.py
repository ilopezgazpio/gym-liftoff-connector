import torch
import pyautogui
from gym_liftoff.envs.telemetry import init_udp_socket
from gym_liftoff.envs.detector import CrashDetector
from gym_liftoff.main import VideoSampler
import socket
import struct
import numpy as np
#import zarr
#from numcodecs import Blosc
from queue import Queue
from .EpisodeWriterThread import EpisodeWriterThread

class RecordingTool:
    def __init__(self, zarr_path):
        self.sock = init_udp_socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256)
        self.reading_size = int(self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
        self.sc_w = self.sc_h = 256
        self.crash_detector = CrashDetector()
        self.resetting = False
        self.video_sampler = VideoSampler.VideoSampler(self.sc_w, self.sc_h)
        self.zarr_path = zarr_path
        #self.root = zarr.open(self.zarr_path, mode="a")

    def get_info(self):
        return self.read_telemetry()

    def reset(self):
        pyautogui.press('r')

    def start(self, max_time_recording, ep_id):
        self.reset()
        max_time = 86400*max_time_recording[0] + 3600*max_time_recording[1] + 60 * max_time_recording[2] + max_time_recording[3]
        current_time = starting_time = self.read_telemetry()["timestamp"]

        queue = Queue(maxsize=1024)

        ep = self.create_episode(ep_id)
        writer_thread = EpisodeWriterThread(ep, queue, chunk_size=32)
        writer_thread.start()
        done = False
        while current_time < max_time + starting_time and not done:
            obs = self.video_sampler.sample()
            info = self.get_info()
            current_time = info["timestamp"]
            done = self.__terminated__(info)
            info_conc = np.concatenate([np.array(info["timestamp"]), info["position"], info["attitude"], info["velocity"], info["gyro"], info["input"]])
            queue.put((obs, info_conc, done))

        self.reset()

    def __terminated__(self, info):
        return self.crash_detector.is_crashed(info)


    def read_telemetry(self):
        latest = None
        _ = self.sock.recvfrom(self.reading_size)
        while True:
            try:
                data, _ = self.sock.recvfrom(128)  # leer todo lo disponible
                latest = data
                break
            except BlockingIOError:
                continue

        if latest is None:
            return None

        if len(latest) < 72:
            return None

        unpacked = struct.unpack('18f', latest[:72])

        timestamp = unpacked[0]
        pos = np.array(unpacked[1:4])
        att = np.array(unpacked[4:8])
        vel = np.array(unpacked[8:11])
        gyro = np.array(unpacked[11:14])
        inp = np.array(unpacked[14:18])

        return {
            'timestamp': timestamp,
            'position': pos,
            'attitude': att,
            'velocity': vel,
            'gyro': gyro,
            'input': inp
        }

    def create_episode(self, ep_id):
        ep = self.root.require_group(f"episodes/{ep_id:06d}")
        ep.attrs["length"] = 0
        ep.create_dataset(
            "obs",
            shape=(None, 3, 256, 256),
            chunks=(32, 3, 256, 256),
            dtype="float32",
            compressor=Blosc(cname='zstd', clevel=3)
        )
        ep.create_dataset(
            "info",
            shape=(None, 18),
            chunks=(32, 18),
            dtype="float32"
        )
        ep.create_dataset(
            "dones",
            shape=(None,),
            chunks=(32,),
            dtype="uint8"
        )
        return ep