import lmdb
import pickle
from queue import Queue
from threading import Thread

class LMDBWriter:
    def __init__(self, lmdb_path, batch_size=32, queue_size=500, map_size=20*1024**3):
        self.env = lmdb.open(lmdb_path, map_size=map_size)
        self.queue = Queue(maxsize=queue_size)
        self.batch_size = batch_size
        self.idx = 0
        self.thread = Thread(target=self._writer_thread, daemon=True)
        self.thread.start()
        self.closed = False

    def _writer_thread(self):
        while True:
            batch = []

            while len(batch) < self.batch_size:
                item = self.queue.get()
                if item is None:  # final signal
                    break
                batch.append(item)
                self.queue.task_done()

            if not batch:
                break

            with self.env.begin(write=True) as txn:
                for sample in batch:
                    txn.put(f"{self.idx:08d}".encode(), pickle.dumps(sample))
                    self.idx += 1

            del batch

    def put(self, sample):
        if self.closed:
            raise RuntimeError("Writer already closed")
        self.queue.put(sample)

    def close(self):
        if self.closed:
            return
        self.queue.put(None)      # final signal
        self.thread.join()
        self.env.close()
        self.closed = True