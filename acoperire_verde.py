import os
import cv2
import numpy as np

def calculate_green_coverage_for_folder(images_folder, labels_folder):
    results = []

    # Iterează prin toate imaginile din folder
    for image_file in os.listdir(images_folder):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):  # Verifică formatul imaginii
            image_path = os.path.join(images_folder, image_file)

            # Asociază fișierul `.txt` pentru imagine
            label_file = os.path.splitext(image_file)[0] + ".txt"
            labels_path = os.path.join(labels_folder, label_file)

            # Verifică dacă fișierul `.txt` există
            if os.path.exists(labels_path):
                img = cv2.imread(image_path)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # Filtru pentru "verde"
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])
                mask = cv2.inRange(hsv, lower_green, upper_green)

                # Calcul procentaj verde total
                total_pixels = img.shape[0] * img.shape[1]
                green_pixels = cv2.countNonZero(mask)
                total_green_percentage = (green_pixels / total_pixels) * 100

                # Calcul procentaj verde în bounding boxes
                green_in_boxes = 0
                total_in_boxes = 0

                with open(labels_path, 'r') as f:
                    boxes = f.readlines()

                for box in boxes:
                    elements = box.strip().split()
                    x_center, y_center, width, height = map(float, elements[1:])

                    # Conversie coordonate YOLO -> OpenCV
                    img_h, img_w, _ = img.shape
                    x1 = int((x_center - width / 2) * img_w)
                    y1 = int((y_center - height / 2) * img_h)
                    x2 = int((x_center + width / 2) * img_w)
                    y2 = int((y_center + height / 2) * img_h)

                    # Decupează bounding box
                    box_mask = mask[y1:y2, x1:x2]
                    green_in_boxes += cv2.countNonZero(box_mask)
                    total_in_boxes += (x2 - x1) * (y2 - y1)

                green_coverage_in_boxes = (green_in_boxes / total_in_boxes) * 100 if total_in_boxes > 0 else 0

                results.append({
                    "image": image_file,
                    "total_green_percentage": total_green_percentage,
                    "green_coverage_in_boxes": green_coverage_in_boxes,
                })

                print(f"Imagine: {image_file} | Verde total: {total_green_percentage:.2f}% | Verde în bounding boxes: {green_coverage_in_boxes:.2f}%")
            else:
                print(f"Fișierul .txt nu există pentru: {image_file}")

    return results

# Exemplu de rulare
images_folder = "/Users/mateimaat/Desktop/data/sorghumfield.v3-416x416_augmented.yolov5pytorch/test/images"  # Folderul cu imagini
labels_folder = "/Users/mateimaat/Desktop/yolo_results/bound_plant_row"  # Folderul cu fișierele .txt combinate

results = calculate_green_coverage_for_folder(images_folder, labels_folder)

# Salvează rezultatele într-un fișier CSV
import pandas as pd
results_df = pd.DataFrame(results)
results_df.to_csv("/Users/mateimaat/Desktop/yolo_results/green_coverage_results.csv", index=False)
print("Rezultatele au fost salvate în green_coverage_results.csv")