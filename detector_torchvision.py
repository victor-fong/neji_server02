import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
import torchvision
import argparse
import time
import logging
import numpy as np
from threading import Thread

from status_worker import StatusWorker
from recent_queue import RecentQueue

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

COLORS = np.random.uniform(0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))

transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    with torch.no_grad():
        start_time = time.time()
        outputs = model(image) # get the predictions on the image
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        logging.debug(f"Detection time: {duration}ms")

    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > detection_threshold]
    # get all the predicted bounding boxes
    bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = bboxes[np.array(scores) >= detection_threshold].astype(np.int32)
    # get all the predicited class names
    labels = outputs[0]['labels'].cpu().numpy()
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[labels[i]] for i in thresholded_preds_inidices]
    return boxes, pred_classes

def draw_boxes(boxes, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[COCO_INSTANCE_CATEGORY_NAMES.index(classes[i])]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image

class Detector:
    def __init__(self, from_queue, status_worker, should_save=True, max_save=50):
        self.model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights='DEFAULT')
        # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'

        self.model.eval().to(self.device)

        logging.info(f"Object Detection Initialized on Device {self.device}")

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
            boxes, classes = predict(img, self.model, self.device, 0.3)

            if self.should_save:
                result = draw_boxes(boxes, classes, img)
                jpg_filename = f"work/detected_frame{counter:03d}.jpg"
                cv2.imwrite(jpg_filename, result)
            self.status_worker.inc_detected()

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    from_queue = RecentQueue(1)
    status_worker = StatusWorker()
    status_worker.start()
    processor = Detector(from_queue, status_worker)
    for i in range(1):
        processor.start()
    img = cv2.imread("examples/frame01.jpg", cv2.IMREAD_COLOR)
    while True:
        from_queue.enqueue(img)
        status_worker.inc_processed()
        time.sleep(0.1)
