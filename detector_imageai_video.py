from imageai.Detection import VideoObjectDetection
import time
import logging
import numpy as np
from threading import Thread
import cv2

from fake_camera import FakeCamera
from status_worker import StatusWorker
from recent_queue import RecentQueue

class Detector:
    def __init__(self, camera, status_worker, should_save=True, max_save=50):

        self.detector = VideoObjectDetection()
        # self.detector.setModelTypeAsYOLOv3()
        # self.detector.setModelPath("models/yolo.h5")

        # self.detector.setModelTypeAsTinyYOLOv3()
        # self.detector.setModelPath("models/yolo-tiny.h5")

        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath("models/resnet50_coco_best_v2.1.0.h5")



        # self.detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))


        self.detector.loadModel()
        # self.detector.detectObjectsFromImage(input_image="examples/frame01.jpg", input_type='file', output_type='array', minimum_percentage_probability=30)

        self.status_worker = status_worker
        self.camera = camera
        # self.from_queue = from_queue
        self.should_save = should_save
        self.max_save = max_save



    def start(self):
        thread = Thread(target=self.detect_from_camera)
        thread.start()

    def per_frame_function(self, frame_number, output_array, output_count):
        # logging.debug(f"Finished frame {frame_number}")
        # if self.should_save:
        #     jpg_filename = f"work/detected_frame{frame_number:03d}.jpg"
        #     cv2.imwrite(jpg_filename, output_array)
        self.status_worker.inc_detected()

    def detect_from_camera(self):
        self.detector.detectObjectsFromVideo(camera_input=self.camera, save_detected_video=False,
                        frames_per_second=20,
                        per_frame_function=self.per_frame_function,
                        log_progress=False, minimum_percentage_probability=20)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
    from_queue = RecentQueue(1)
    fake_camera = FakeCamera(from_queue)
    status_worker = StatusWorker()
    processor = Detector(fake_camera, status_worker)
    status_worker.start()
    for i in range(1):
        processor.start()
    img = cv2.imread("examples/frame01.jpg", cv2.IMREAD_COLOR)
    while True:
        from_queue.enqueue(img)
        status_worker.inc_processed()
        time.sleep(0.1)
