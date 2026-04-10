import lmdb
import pickle
from queue import Queue
from threading import Thread

class LMDBWriter:
    def __init__(self, lmdb_path, max_size = 50000, batch_size=32, queue_size=500, map_size=16*1024**3, replay_buffer = False):
        self.lmdb_path = lmdb_path
        self.queue_size = queue_size
        self.max_size = max_size
        self.map_size = map_size
        self.batch_size = batch_size
        self.idx = 0
        self.closed = True
        self.replay_buffer = replay_buffer
        #self.open()

    def _writer_thread(self):
        while True:
            batch = []
            while len(batch) < self.batch_size:
                item = self.queue.get()
                if item is None:
                    self.queue.task_done()
                    break
                batch.append(item)
                self.queue.task_done()

            if batch:
                with self.env.begin(write=True) as txn:
                    for sample in batch:
                        real_idx = self.idx % self.max_size
                        txn.put(f"{real_idx:08d}".encode(), pickle.dumps(sample))
                        self.idx += 1

            if self.replay_buffer:
                self.size = self.size + len(batch) if self.size < self.max_size else self.max_size
                txn.put(b"__idx__", pickle.dumps(self.idx))
                txn.put(b"__size__", pickle.dumps(self.size))

            if item is None:
                break  # salida del hilo

    def put(self, sample):
        if self.closed:
            raise RuntimeError("Writer already closed")
        self.queue.put(sample)

    def open(self):
        self.closed = False
        self.env = lmdb.open(self.lmdb_path, map_size=self.map_size)
        with self.env.begin() as txn:
            raw = txn.get(b"__idx__")
            if raw:
                self.idx = pickle.loads(raw)
            else:
                self.idx = txn.stat()['entries']

            raw_size = txn.get(b"__size__")
            if raw_size:
                self.size = pickle.loads(raw_size)
            else:
                self.size = 0

        self.queue = Queue(maxsize=self.queue_size)
        self.thread = Thread(target=self._writer_thread, daemon=True)
        self.thread.start()

    def clear_database(self):
        self.queue.join()
        with self.env.begin(write=True) as txn:
            default_db = self.env.open_db()
            txn.drop(db=default_db, delete=False)
        self.idx = 0
        self.size = 0

    def close(self):
        if self.closed:
            return
        self.queue.put(None)  # señal de fin
        self.thread.join()
        self.env.close()
        self.closed = True