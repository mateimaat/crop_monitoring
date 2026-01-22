import cv2
import numpy as np

def draw_yolo_boxes(image_path, txt_path, window_name):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, xc, yc, bw, bh = map(float, parts)

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow(window_name, img)

# === SETEAZĂ CALEA TA AICI ===
image_path = "/Users/mateimaat/Desktop/data/sorghumfield.v3-416x416_augmented.yolov5pytorch/test/images/training_124_png.rf.880cf55a745bd87522d9ac1cf09334be.jpg"
txt_path_1 = "/Users/mateimaat/Desktop/Metoda_HSV/labels_pipeline7/training_124_png.rf.880cf55a745bd87522d9ac1cf09334be.txt"
txt_path_2 = "/Users/mateimaat/Desktop/Metoda_HSV/labels_pipeline_GR_1/training_124_png.rf.880cf55a745bd87522d9ac1cf09334be.txt"

# Desenează și afișează
draw_yolo_boxes(image_path, txt_path_1, "Model culoare HSV")
draw_yolo_boxes(image_path, txt_path_2, "Model culoare G-R")

cv2.waitKey(0)
cv2.destroyAllWindows()