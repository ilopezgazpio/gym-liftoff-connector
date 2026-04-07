import threading
import numpy as np

class EpisodeWriterThread(threading.Thread):
    def __init__(self, ep, queue, chunk_size=32):
        super().__init__()
        self.ep = ep
        self.queue = queue
        self.chunk_size = chunk_size
        self.obs_buffer = []
        self.info_buffer = []
        self.done_buffer = []
        self.stop_signal = False

    def run(self):
        while not self.stop_signal or not self.queue.empty():
            try:
                img, state, done = self.queue.get(timeout=0.1)
                self.obs_buffer.append(img)
                self.info_buffer.append(state)
                self.done_buffer.append(done)

                if len(self.obs_buffer) >= self.chunk_size:
                    self.insert()
            except:
                continue
        self.insert()

    def insert(self):
        if not self.obs_buffer:
            return
        n = self.ep.attrs["length"]
        k = len(self.obs_buffer)
        self.ep["obs"].resize(n+k, axis=0)
        self.ep["info"].resize(n+k, axis=0)
        self.ep["dones"].resize(n+k, axis=0)

        self.ep["images"][n:n+k] = np.stack(self.obs_buffer)
        self.ep["states"][n:n+k] = np.stack(self.info_buffer)
        self.ep["dones"][n:n+k] = np.array(self.done_buffer, dtype=np.uint8)

        self.ep.attrs["length"] += k
        self.obs_buffer.clear()
        self.info_buffer.clear()
        self.done_buffer.clear()

    def stop(self):
        self.stop_signal = True