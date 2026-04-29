import lmdb
import pickle
from queue import Queue
from threading import Thread

class LMDBWriter:
    def __init__(self, lmdb_path, max_size = 50000, batch_size=32, queue_size=500, map_size=16*1024**3, replay_buffer = False, pickle_save = True):
        self.lmdb_path = lmdb_path
        self.queue_size = queue_size
        self.max_size = max_size
        self.map_size = map_size
        self.batch_size = batch_size
        self.idx = 0
        self.closed = True
        self.replay_buffer = replay_buffer
        self.pickle_save = pickle_save
        #self.open()

    def _writer_thread(self):
        while True:
            while len(self.batch) < self.batch_size:
                item, idx = self.queue.get()
                if item is None:
                    self.queue.task_done()
                    break
                self.batch.append(item)
                self.ids.append(idx)
                self.queue.task_done()

            if self.batch:
                self.flush()

            if self.replay_buffer:
                with self.env.begin(write=True) as txn:
                    self.size = self.idx if self.idx < self.max_size else self.max_size
                    txn.put(b"__idx__", pickle.dumps(self.idx))
                    txn.put(b"__size__", pickle.dumps(self.size))

            if item is None:
                break  # salida del hilo

    def put(self, sample, idx = None):
        if self.closed:
            raise RuntimeError("Writer already closed")
        self.queue.put((sample, idx))

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
        self.batch = []
        self.ids = []
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
        self.queue.put((None, None))  # señal de fin
        self.thread.join()
        self.env.close()
        self.closed = True

    def flush(self):
        with self.env.begin(write=True) as txn:
            for i, sample in zip(self.ids, self.batch):
                if i == None:
                    real_idx = self.idx % self.max_size
                    self.idx += 1
                else:
                    real_idx = i

                if self.pickle_save:
                    pack = pickle.dumps(sample)
                else:
                    pack = sample.tobytes()
                txn.put(f"{real_idx:08d}".encode(), pack)

        self.batch = []
        self.ids = []

    def get_item(self, idx):
        with self.env.begin() as txn:
            data = txn.get(idx)
        return data






