import threading
import numpy as np
import gzip
from status_worker import StatusWorker
from recent_queue import RecentQueue
import cv2


class FrameProcessor:
    def __init__(self, name, from_queue, to_queue, status_worker, should_save=True, max_save=50):
        self.name = name
        self.from_queue = from_queue
        self.to_queue = to_queue
        self.status_worker = status_worker
        self.should_save = should_save
        self.max_save = max_save

    def start(self):
        processor_thread = threading.Thread(target=self.process_from_queue)
        processor_thread.start()

    def YUVtoRGB(self, byteArray, w, h):
        byteArray = np.frombuffer(byteArray, dtype=np.ubyte)
        e = w*h
        Y = byteArray[0:e]
        Y = np.reshape(Y, (h,w))
        # Y = Y[:,:1920]

        s = e + int(e/4)
        V = byteArray[e:s]
        V = np.repeat(V, 2, 0)
        V = np.reshape(V, (int(h/2),w))
        V = np.repeat(V, 2, 0)
        # V = V[:,:1920]

        U = byteArray[s:]
        U = np.repeat(U, 2, 0)
        U = np.reshape(U, (int(h/2),w))
        U = np.repeat(U, 2, 0)
        # U = U[:,:1920]

        RGBMatrix = (np.dstack([Y,U,V])).astype(np.uint8)
        RGBMatrix = cv2.cvtColor(RGBMatrix, cv2.COLOR_YUV2RGB, 3)
        return RGBMatrix

    def decompress(self, buf):
        newbuf = gzip.decompress(buf)
        return newbuf

    def process_from_queue(self):
        counter = 0
        while True:
            counter += 1
            if counter > self.max_save:
                counter = 1

            buf = self.from_queue.dequeue()

            buf = self.decompress(buf)
            w = int.from_bytes(buf[0:4], "big")
            h = int.from_bytes(buf[4:8], "big")
            buf = buf[8:]

            img = self.YUVtoRGB(buf, w, h)
            if self.should_save:
                jpg_filename = f"work/processed_frame_{self.name}_{counter:03d}.jpg"
                cv2.imwrite(jpg_filename, img)
                detected_filename = f"work/detected_frame{counter:03d}.jpg"
            self.to_queue.enqueue(img)
            self.status_worker.inc_processed()

if __name__ == "__main__":
    from_queue = RecentQueue(1)
    to_queue = RecentQueue(1)
    status_worker = StatusWorker()
    processor = FrameProcessor("worker1", from_queue, to_queue, status_worker)
    processor.start()
    with open("examples/recv_frame001.dat", "rb") as file:
        buf = file.read()
        from_queue.enqueue(buf)
