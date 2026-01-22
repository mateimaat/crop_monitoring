import cv2
import pandas as pd
import numpy as np

# Căile către fișier și imagine
csv_path = "/Users/mateimaat/Desktop/Green Masks/boundboxes_randuri15.csv"
image_path = "/Users/mateimaat/Desktop/data/sorghumfield.v3-416x416_augmented.yolov5pytorch/valid/images/validation_04_png.rf.12ab1f8625c0b6d9c7caadcffb08483c.jpg"

# Încarcă imaginea
img = cv2.imread(image_path)
h, w, _ = img.shape
total_pixels = h * w

# Convertire HSV și mască pentru verde
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_green = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([80, 255, 255]))
green_pixels_total = cv2.countNonZero(mask_green)
green_percent_total = (green_pixels_total / total_pixels) * 100

# Încarcă bounding box-urile din CSV
df = pd.read_csv(csv_path)

# Calculează verdele în interiorul rândurilor
green_in_rows = 0
pixels_in_rows = 0

for _, row in df.iterrows():
    x_center = int(row["x_center"] * w)
    y_center = int(row["y_center"] * h)
    box_width = int(row["width"] * w)
    box_height = int(row["height"] * h)

    x1 = max(x_center - box_width // 2, 0)
    y1 = max(y_center - box_height // 2, 0)
    x2 = min(x_center + box_width // 2, w)
    y2 = min(y_center + box_height // 2, h)

    roi = mask_green[y1:y2, x1:x2]
    green_in_rows += cv2.countNonZero(roi)
    pixels_in_rows += roi.size

green_percent_rows = (green_in_rows / pixels_in_rows) * 100 if pixels_in_rows > 0 else 0

# Afișare rezultate
print(f"Procent verde total: {green_percent_total:.2f}%")
print(f"Procent verde în rânduri: {green_percent_rows:.2f}%")