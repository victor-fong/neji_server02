from threading import Thread, Lock
import time
import logging

TIME_SPAN = 5.0

class StatusWorker:
    def __init__(self):
        self.lock = Lock()
        self.recv = 0
        self.processed = 0
        self.detected = 0

    def start(self):
        thread = Thread(target=self.status_thread)
        thread.start()

    def inc_recv(self):
        with self.lock:
            self.recv += 1

    def inc_processed(self):
        with self.lock:
            self.processed += 1

    def inc_detected(self):
        with self.lock:
            self.detected += 1

    def status_thread(self):
        old_time = round(time.time()*1000)
        while True:
            time.sleep(TIME_SPAN)
            with self.lock:
                new_time = round(time.time()*1000)
                crecv_counter = self.recv
                cprocessed_counter = self.processed
                cdetected_counter = self.detected
                self.recv = 0
                self.processed = 0
                self.detected = 0

            time_span = float(new_time - old_time) / 1000.0
            recv_fps = crecv_counter / time_span
            process_fps = cprocessed_counter / time_span
            detected_fps = cdetected_counter / time_span
            old_time = new_time
            logging.info(f"RECEIVED {recv_fps:5.4f} FPS | PROCESSED {process_fps:5.4f} FPS | DETECTED {detected_fps:5.4f} FPS | TIME SPAN {time_span:5.3f}s")

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    worker = StatusWorker()
    worker.start()
