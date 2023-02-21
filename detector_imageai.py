from imageai.Detection import ObjectDetection
import time
import logging
import numpy as np
from threading import Thread
import cv2

from status_worker import StatusWorker
from recent_queue import RecentQueue

class Detector:
    def __init__(self, from_queue, status_worker, should_save=True, max_save=50):

        self.detector = ObjectDetection()
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath("models/yolo.h5")

        # self.detector.setModelTypeAsTinyYOLOv3()
        # self.detector.setModelPath("models/yolo-tiny.h5")

        # self.detector.setModelTypeAsRetinaNet()
        # self.detector.setModelPath("models/resnet50_coco_best_v2.1.0.h5")



        # self.detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))


        self.detector.loadModel()
        self.detector.detectObjectsFromImage(input_image="examples/frame01.jpg", input_type='file', output_type='array', minimum_percentage_probability=30)

        self.status_worker = status_worker
        self.from_queue = from_queue
        self.should_save = should_save
        self.max_save = max_save



    def start(self):
        thread = Thread(target=self.detect_from_queue)
        thread.start()

    def detect_from_queue(self):
        counter = 0
        while True:
            counter += 1
            if counter > self.max_save:
                counter = 1
            img = self.from_queue.dequeue()
            start_time = time.time()
            if self.should_save:
                detected_filename = f"work/detected_frame{counter:03d}.jpg"
                detections = self.detector.detectObjectsFromImage(input_type="array", input_image=img, output_image_path=detected_filename, output_type="file", minimum_percentage_probability=20, thread_safe=True)
            else:
                returned_image, detections = self.detector.detectObjectsFromImage(input_type="array", input_image=img, output_type="array", minimum_percentage_probability=20, thread_safe=True)
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            logging.debug(f"Detection time: {duration}ms")
            self.status_worker.inc_detected()

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
    from_queue = RecentQueue(1)
    status_worker = StatusWorker()
    processor = Detector(from_queue, status_worker)
    status_worker.start()
    for i in range(1):
        processor.start()
    img = cv2.imread("examples/frame01.jpg", cv2.IMREAD_COLOR)
    while True:
        from_queue.enqueue(img)
        status_worker.inc_processed()
        time.sleep(0.1)
