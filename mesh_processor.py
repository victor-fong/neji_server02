import threading
import numpy as np
import gzip
from all_status_worker import AllStatusWorker
from recent_queue import RecentQueue
import cv2
import logging
import struct


class MeshProcessor:
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
            logging.debug(f"Before Decompress {len(buf)} bytes")
            buf = self.decompress(buf)
            logging.debug(f"After Decompress {len(buf)} bytes")

            i = 0
            mesh_id_length = int.from_bytes(buf[i:i+4], "little")
            i += 4
            logging.debug(f"Received mesh with {mesh_id_length} bytes ID")
            mesh_id = buf[i:i+mesh_id_length].decode('ASCII')
            i += mesh_id_length
            logging.debug(f"Received mesh with ID: {mesh_id}")

            vertices_num = int.from_bytes(buf[i:i+4], "little")
            i += 4
            logging.debug(f"Received mesh {vertices_num} vertices")

            for j in range(vertices_num):
                x = struct.unpack('<f', buf[i : i+4])[0]
                i+=4
                y = struct.unpack('<f', buf[i : i+4])[0]
                i+=4
                z = struct.unpack('<f', buf[i : i+4])[0]
                i+=4
                logging.debug(f"VERT {j} | X {x:4.6f} | Y {y:4.6f} | Z {z:4.6f}")
            logging.debug(f"Read {i} bytes in total")
            #
            # if self.should_save:
            #     jpg_filename = f"work/meshes/processed_mesh_{self.name}_{counter:03d}.jpg"
            #     cv2.imwrite(jpg_filename, img)
            # self.to_queue.enqueue()
            self.status_worker.inc("PROCESSED_MESH")

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
    from_queue = RecentQueue(1)
    to_queue = RecentQueue(1)
    status_worker = AllStatusWorker()
    processor = MeshProcessor("worker1", from_queue, to_queue, status_worker)
    processor.start()
    with open("examples/recv_mesh001.dat", "rb") as file:
        buf = file.read()
        from_queue.enqueue(buf)
