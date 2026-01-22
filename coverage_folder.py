import os
import cv2
import numpy as np

def calculate_green_coverage(images_folder, labels_folder, output_folder):
    # Creează folderul de output dacă nu există
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in os.listdir(images_folder):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(images_folder, image_file)
            label_file = os.path.join(labels_folder, image_file.replace(".jpg", ".txt"))

            if not os.path.exists(label_file):
                print(f"Fișierul de etichete lipsește pentru {image_file}. Se trece mai departe.")
                continue

            # Citește imaginea
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
            with open(label_file, 'r') as f:
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

            # Scrie rezultatele într-un fișier
            output_file = os.path.join(output_folder, image_file.replace(".jpg", ".txt"))
            with open(output_file, 'w') as f:
                f.write(f"Total Green Coverage: {total_green_percentage:.2f}%\n")
                f.write(f"Green Coverage in Bounding Boxes: {green_coverage_in_boxes:.2f}%\n")

            print(f"Acoperirea verde pentru {image_file} a fost salvată în {output_file}")

images_folder = "/Users/mateimaat/Desktop/data/sorghumfield.v3-416x416_augmented.yolov5pytorch/test/images"  # Folderul cu imaginile
labels_folder = "/Users/mateimaat/Desktop/data/sorghumfield.v3-416x416_augmented.yolov5pytorch/test/labels"  # Folderul cu fișierele de etichete
output_folder = "/Users/mateimaat/Desktop/yolo_results/coverage_data"  # Folderul pentru datele de acoperire verde

calculate_green_coverage(images_folder, labels_folder, output_folder)