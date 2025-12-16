import torch
import cv2
import os

# Load YOLOv5 pretrained
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_clothes(image_path):
    img = cv2.imread(image_path)
    results = model(img)

    detections = []
    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        if label in ['person', 'shirt', 'pants', 'jacket', 'dress', 'skirt']:
            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]
            detections.append({
                'label': label,
                'image': crop
            })
    return detections
