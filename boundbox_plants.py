import cv2
import os

def visualize_bounding_boxes(image_path, txt_path, output_path):
    # Citește imaginea
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Verifică dacă fișierul .txt există
    if not os.path.exists(txt_path):
        print(f"Fișierul {txt_path} nu există!")
        return

    # Citește coordonatele din fișierul .txt
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        x_center, y_center, bbox_width, bbox_height = map(float, data[1:])

        # Conversie coordonate normalizate la pixeli
        x1 = int((x_center - bbox_width / 2) * w)
        y1 = int((y_center - bbox_height / 2) * h)
        x2 = int((x_center + bbox_width / 2) * w)
        y2 = int((y_center + bbox_height / 2) * h)

        # Alege culoarea în funcție de clasă
        color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
        label = "Row" if class_id == 0 else "Plant"

        # Desenează bounding box și eticheta
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Salvează imaginea cu bounding boxes
    os.makedirs(output_path, exist_ok=True)
    output_image_path = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)
    print(f"Imaginea procesată a fost salvată în {output_image_path}")

# Exemplu de utilizare
image_folder = "/Users/mateimaat/Desktop/data/sorghumfield.v3-416x416_augmented.yolov5pytorch/test/images"
txt_folder = "/Users/mateimaat/Desktop/yolo_results/boundboxes_plants"
output_folder = "/Users/mateimaat/Desktop/yolo_results/plants_bounded_1"

os.makedirs(output_folder, exist_ok=True)

for image_file in os.listdir(image_folder):
    if image_file.endswith((".jpg", ".png")):
        image_path = os.path.join(image_folder, image_file)
        txt_path = os.path.join(txt_folder, os.path.splitext(image_file)[0] + ".txt")
        visualize_bounding_boxes(image_path, txt_path, output_folder)