import cv2
import numpy as np
import os
import time


start_total = time.time()
def calculate_green_coverage_with_bboxes(image_path, bbox_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    try:
        with open(bbox_path, "r") as f:
            bboxes = [line.strip().split() for line in f.readlines()]
    except FileNotFoundError:
        print(f"[WARNING] Lipsă fișier de etichete pentru: {image_path}")
        return None

    total_green_pixels = 0
    total_pixels = 0

    for bbox in bboxes:
        class_id, x_center, y_center, width, height = map(float, bbox)
        h, w = img.shape[:2]

        x_center, y_center = int(x_center * w), int(y_center * h)
        width, height = int(width * w), int(height * h)

        x1 = max(0, int(x_center - width / 2))
        y1 = max(0, int(y_center - height / 2))
        x2 = min(w, int(x_center + width / 2))
        y2 = min(h, int(y_center + height / 2))

        roi = hsv[y1:y2, x1:x2]
        mask = cv2.inRange(roi, lower_green, upper_green)
        green_pixels = cv2.countNonZero(mask)
        roi_pixels = roi.shape[0] * roi.shape[1]

        total_green_pixels += green_pixels
        total_pixels += roi_pixels

    return (total_green_pixels / total_pixels) * 100 if total_pixels > 0 else 0


image_dir = "/Users/mateimaat/Desktop/DEMO/images"
label_dir = "/Users/mateimaat/Desktop/DEMO/labels_pipeline"

results = []

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        label_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)

        coverage = calculate_green_coverage_with_bboxes(image_path, label_path)
        if coverage is not None:
            results.append((filename, coverage))
            print(f"{filename} -> Grad acoperire verde: {coverage:.2f}%")

end_total = time.time()
print(f"Timp: {end_total - start_total:.2f} secunde.")
