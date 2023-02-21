import socket
import threading
import logging
from all_status_worker import AllStatusWorker
from recent_queue import RecentQueue

def display_transmission(key, data, time_span):
    mbps = data / 1000000 / time_span
    return f"{key}: {mbps:4.4f} MBps | "

def display_fps(key, data, time_span):
    fps = data / time_span
    return f"{key}: {fps:4.4f} FPS | "

class RecvSocket:
    def __init__(self, status_worker, to_queue, port=8989, host="0.0.0.0", should_save=True, max_save=50):
        self.host = host
        self.port = port
        self.to_queue = to_queue
        self.status_worker = status_worker
        self.should_save = should_save
        self.max_save = max_save

    def start(self):
        socket_thread = threading.Thread(target=self.listen)
        socket_thread.start()

    def listen(self):
        counter = 0;
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.host, self.port))
                s.listen()
                logging.info("Waiting for Connection...")
                conn, addr = s.accept()
                with conn:
                    try:
                        logging.debug(f"Connected by {addr}")
                        while True:
                            counter += 1
                            if counter > self.max_save:
                                counter = 1

                            data = conn.recv(4)
                            if not data:
                                break
                            frameLength = int.from_bytes(data, "big")
                            logging.debug(f"Receiving {frameLength} bytes of frame data")
                            chunks = []
                            bytes_read = 0
                            while bytes_read < frameLength:
                                chunk = conn.recv(min(frameLength - bytes_read, 4096))
                                if chunk == b'':
                                    raise RuntimeError("socket connection broken")
                                chunks.append(chunk)
                                bytes_read = bytes_read + len(chunk)
                            chunks = b''.join(chunks)

                            if self.should_save:
                                file = open(f"work/recv_frame{counter:03d}.dat", "wb")
                                file.write(chunks)
                                file.close()

                            next_frame = chunks
                            self.to_queue.enqueue(next_frame)
                            self.status_worker.inc("RECV", count=frameLength+4, display_method=display_transmission)
                            self.status_worker.inc("RECV_FRAME", display_method=display_fps)

                    except:
                        logging.exception("")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    to_queue = RecentQueue(1)
    status_worker = AllStatusWorker()
    status_worker.start()
    recv_socket = RecvSocket(status_worker, to_queue)
    recv_socket.start()
