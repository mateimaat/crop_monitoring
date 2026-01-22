import cv2
import numpy as np
import os

def load_yolo_boxes(txt_path, img_shape):
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, xc, yc, bw, bh = map(float, parts)
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask

# Înlocuiește cu path-urile tale:
txt_yolo = "/Users/mateimaat/Desktop/yolo_results/exp2/labels/testing_35_png.rf.d5391a0424a348be3bef0010be20801f.txt"
txt_model = "/Users/mateimaat/Desktop/Metoda_HSV/labels_pipeline/testing_35_png.rf.d5391a0424a348be3bef0010be20801f.txt"
image_size = (416, 416)  # dimensiunea imaginilor

mask_yolo = load_yolo_boxes(txt_yolo, image_size)
mask_model = load_yolo_boxes(txt_model, image_size)

intersection = cv2.bitwise_and(mask_yolo, mask_model)
union = cv2.bitwise_or(mask_yolo, mask_model)

intersection_area = np.sum(intersection == 255)
union_area = np.sum(union == 255)
iou_score = intersection_area / union_area if union_area > 0 else 0.0

print(f"IoU (pixel-level mask): {iou_score:.4f}")