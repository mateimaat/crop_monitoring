import cv2
import numpy as np
import os

def generate_individual_labels(image_path, output_txt_path, class_id=1):
    # Citește imaginea
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definește intervalul pentru verde
    lower_green = np.array([35, 40, 40])  # Ajustează dacă este necesar
    upper_green = np.array([85, 255, 255])

    # Creează o mască pentru zonele verzi
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Detectează contururile
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = image.shape[:2]
    lines = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        # Normalizează coordonatele pentru YOLO
        x_center = (x + width / 2) / w
        y_center = (y + height / 2) / h
        norm_width = width / w
        norm_height = height / h

        # Adaugă doar bounding boxes suficient de mari
        if norm_width > 0.01 and norm_height > 0.01:
            lines.append(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")

    # Scrie în fișierul .txt
    with open(output_txt_path, 'w') as f:
        f.writelines(lines)

# Exemplu de utilizare
image_folder = "/Users/mateimaat/Desktop/data/sorghumfield.v3-416x416_augmented.yolov5pytorch/test/images"
output_folder = "/Users/mateimaat/Desktop/yolo_results/boundboxes_plants"
os.makedirs(output_folder, exist_ok=True)

for image_file in os.listdir(image_folder):
    if image_file.endswith((".jpg", ".png")):
        image_path = os.path.join(image_folder, image_file)
        output_txt_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + ".txt")
        generate_individual_labels(image_path, output_txt_path)