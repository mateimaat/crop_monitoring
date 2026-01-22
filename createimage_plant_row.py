import cv2
import os

def process_all_images(images_folder, labels_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Iterează prin toate imaginile din folder
    for image_file in os.listdir(images_folder):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):  # Verifică formatul imaginii
            image_path = os.path.join(images_folder, image_file)

            # Asociază fișierul `.txt` pentru imagine
            label_file = os.path.splitext(image_file)[0] + ".txt"
            labels_path = os.path.join(labels_folder, label_file)

            # Verifică dacă fișierul `.txt` există
            if os.path.exists(labels_path):
                # Desenează bounding boxes
                img = cv2.imread(image_path)
                with open(labels_path, 'r') as f:
                    boxes = f.readlines()

                for box in boxes:
                    elements = box.strip().split()
                    label = int(elements[0])  # Clasa
                    x_center, y_center, width, height = map(float, elements[1:])

                    # Conversie coordonate
                    img_h, img_w, _ = img.shape
                    x1 = int((x_center - width / 2) * img_w)
                    y1 = int((y_center - height / 2) * img_h)
                    x2 = int((x_center + width / 2) * img_w)
                    y2 = int((y_center + height / 2) * img_h)

                    # Desenează bounding box
                    color = (0, 255, 0) if label == 0 else (255, 0, 0)  # Verde pentru rânduri, roșu pentru plante
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                

                # Salvaree
                output_path = os.path.join(output_folder, image_file)
                cv2.imwrite(output_path, img)
                print(f"Imagine procesată salvată în: {output_path}")
            else:
                print(f"Fișierul .txt nu există pentru: {image_file}")

# Exemplu de rulare
images_folder = "/Users/mateimaat/Desktop/DEMO/images"  # Folderul cu imagini
labels_folder = "/Users/mateimaat/Desktop/DEMO/labels_pipeline"  # Folderul cu fișierele .txt combinate
output_folder = "/Users/mateimaat/Desktop/DEMO/imagini_prelucrare"  # Folderul de output

process_all_images(images_folder, labels_folder, output_folder)