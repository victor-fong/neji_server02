import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
import torchvision
import argparse
import numpy as np

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
        outputs = model(image) # get the predictions on the image
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


# model = torchvision.models.detection.retinanet_resnet50_fpn(weights='RetinaNet_ResNet50_FPN_Weights.COCO_V1')
model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights='DEFAULT')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the model onto the computation device
model.eval().to(device)

# img = cv2.imread("examples/processed_frame001.jpg", cv2.IMREAD_COLOR)
img = cv2.imread("examples/frame01.jpg", cv2.IMREAD_COLOR)
# image_array = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

boxes, classes = predict(img, model, device, 0.2)
print(f"Found {len(classes)} objects")
# get the final image
result = draw_boxes(boxes, classes, img)

cv2.imshow('Image', result)
cv2.waitKey(0)
cv2.imwrite(f"examples/tv_example001.jpg", result)
