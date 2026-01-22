import cv2
import numpy as np
import os
import csv

def load_yolo_boxes(txt_path, img_shape=(416, 416)):
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if not os.path.exists(txt_path):
        return mask

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

labels_folder = "/Users/mateimaat/Desktop/DEMO/labels"
model_folder = "/Users/mateimaat/Desktop/DEMO/labels_pipeline"
output_csv = "/Users/mateimaat/Desktop/DEMO/scor_IoU_model_culoare"

all_files = sorted([f for f in os.listdir(labels_folder) if f.endswith(".txt")])
results = []

for filename in all_files:
    yolo_path = os.path.join(labels_folder, filename)
    model_path = os.path.join(model_folder, filename)

    mask_yolo = load_yolo_boxes(yolo_path)
    mask_model = load_yolo_boxes(model_path)

    intersection = cv2.bitwise_and(mask_yolo, mask_model)
    union = cv2.bitwise_or(mask_yolo, mask_model)

    inter_area = np.sum(intersection == 255)
    union_area = np.sum(union == 255)

    if union_area > 0:
        iou = inter_area / union_area
    else:
        iou = 0.0
    results.append((filename, round(iou, 4)))

mean_iou = round(np.mean([r[1] for r in results]), 4)

# Salvare CSV
with open(output_csv, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Poza", "IoU"])
    writer.writerows(results)
    writer.writerow(["Medie", mean_iou])

print(f"Fisier: {output_csv}")