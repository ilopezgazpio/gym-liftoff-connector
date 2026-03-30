import lmdb
import pickle
from queue import Queue
from threading import Thread

class LMDBWriter:
    def __init__(self, lmdb_path, batch_size=32, queue_size=500, map_size=16*1024**3):
        self.lmdb_path = lmdb_path
        self.queue_size = queue_size
        self.map_size = map_size
        self.batch_size = batch_size
        self.idx = 0
        self.closed = True
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
                        txn.put(f"{self.idx:08d}".encode(), pickle.dumps(sample))
                        self.idx += 1

            if item is None:
                break  # salida del hilo

    def put(self, sample):
        if self.closed:
            raise RuntimeError("Writer already closed")
        self.queue.put(sample)

    def open(self):
        self.closed = False
        self.env = lmdb.open(self.lmdb_path, map_size=self.map_size)
        self.queue = Queue(maxsize=self.queue_size)
        self.thread = Thread(target=self._writer_thread, daemon=True)
        self.thread.start()

    def clear_database(self):
        """
        Vacía toda la base de datos sin cerrar el environment.
        Debe llamarse cuando no hay escritor activo.
        """
        # Esperar a que se vacíe la cola
        self.queue.join()
        with self.env.begin(write=True) as txn:
            default_db = self.env.open_db()
            txn.drop(db=default_db, delete=False)
        self.idx = 0  # reiniciar índice

    def close(self):
        if self.closed:
            return
        self.queue.put(None)  # señal de fin
        self.thread.join()
        self.env.close()
        self.closed = True