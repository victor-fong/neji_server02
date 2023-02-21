from threading import Thread, Lock
import time
import logging

TIME_SPAN = 5.0

def default_display(key, count, time_span):
    result = count / time_span
    return f"{key}: {result:4.4f} | "

class AllStatusWorker:
    def __init__(self):
        self.lock = Lock()
        self.records = {}
        self.should_divide = {}
        self.display_methods = {}

    def start(self):
        thread = Thread(target=self.status_thread)
        thread.start()

    def inc(self, key, should_divide=True, count=1, display_method=default_display):
        with self.lock:
            if key not in self.records.keys():
                self.records[key] = 0
            self.records[key] += count
            self.should_divide[key] = should_divide
            self.display_methods[key] = display_method

    def status_thread(self):
        old_time = round(time.time()*1000)
        while True:
            time.sleep(TIME_SPAN)
            display_row = ""
            with self.lock:
                new_time = round(time.time()*1000)
                time_span = float(new_time - old_time) / 1000.0

                for x in self.records:
                    method = self.display_methods[x]
                    display_row += method(x, self.records[x], time_span)
                    self.records[x] = 0

            display_row += f"TIME_SPAN: {time_span:4.4f}s"
            old_time = new_time
            logging.info(display_row)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    worker = AllStatusWorker()
    worker.start()
    worker.inc("A", should_divide=False)
    worker.inc("B")
    worker.inc("A", should_divide=False)
