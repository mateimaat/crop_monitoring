import cv2
import numpy as np
import os
from scipy.signal import find_peaks
import time

# Foldere
input_folder = "/Users/mateimaat/Desktop/data/sorghumfield.v3-416x416_augmented.yolov5pytorch/test/images"
intermediate_mask_folder = "/Users/mateimaat/Desktop/imagini_detectie/green_masks_GRtimp"
noise_removed_folder = "/Users/mateimaat/Desktop/imagini_detectie/green_masks_noise_removed_GRtimp"
isolated_removed_folder = "/Users/mateimaat/Desktop/imagini_detectie/green_masks_isolated_GRtimp"
final_output_folder = "/Users/mateimaat/Desktop/imagini_detectie/green_masks_boxed_GRtimp"
labels_output_folder = "/Users/mateimaat/Desktop/imagini_detectie/labels_pipeline_GRtimp"

# Asigurare existența folderelor
os.makedirs(intermediate_mask_folder, exist_ok=True)
os.makedirs(noise_removed_folder, exist_ok=True)
os.makedirs(isolated_removed_folder, exist_ok=True)
os.makedirs(final_output_folder, exist_ok=True)
os.makedirs(labels_output_folder, exist_ok=True)

# Parametri
min_area = 5
start_total = time.time()
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    name_only = os.path.splitext(filename)[0]
    image_path = os.path.join(input_folder, filename)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Nu s-a putut încărca imaginea: {filename}")
        continue

    h, w = img.shape[:2]

    # PAS 1 - Detectare verde cu regula G - R > prag
    b, g, r = cv2.split(img)
    prag = 1
    green_mask = np.zeros_like(g, dtype=np.uint8)
    green_mask[(g.astype(int) - r.astype(int)) > prag] = 255
    cv2.imwrite(os.path.join(intermediate_mask_folder, f"{name_only}.png"), green_mask)

    # PAS 2 - Eliminare zgomot
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(green_mask, connectivity=8)
    filtered_mask = np.zeros_like(green_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            filtered_mask[labels == i] = 255
    cv2.imwrite(os.path.join(noise_removed_folder, f"{name_only}.png"), filtered_mask)

    # PAS 3 - Eliminare plante izolate
    cleaned_mask = filtered_mask.copy()
    for x in range(w):
        column = filtered_mask[:, x]
        _, _, stats_col, _ = cv2.connectedComponentsWithStats(column.reshape(-1, 1), connectivity=8)
        if len(stats_col) - 1 < 2:
            cleaned_mask[:, x] = 0
    cv2.imwrite(os.path.join(isolated_removed_folder, f"{name_only}.png"), cleaned_mask)

    # PAS 4 - Proiecție verticală + detectare vârfuri
    vertical_projection = np.sum(cleaned_mask == 255, axis=0)
    peaks, _ = find_peaks(vertical_projection, distance=50, prominence=15)
    if vertical_projection[0] > 100:
        peaks = np.insert(peaks, 0, 0)
    if vertical_projection[-1] > 100:
        peaks = np.append(peaks, len(vertical_projection) - 1)

    result = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
    bboxes = []

    for x in peaks:
        max_width = 30
        left = x
        right = x

        for dx in range(1, max_width):
            if x - dx >= 0 and np.any(cleaned_mask[:, x - dx] == 255):
                left = x - dx
            else:
                break

        for dx in range(1, max_width):
            if x + dx < w and np.any(cleaned_mask[:, x + dx] == 255):
                right = x + dx
            else:
                break

        subregion = cleaned_mask[:, left:right + 1]
        white_coords = np.column_stack(np.where(subregion == 255))
        if len(white_coords) == 0:
            continue

        white_y = white_coords[:, 0]
        white_y_sorted = sorted(white_y)
        gaps = np.diff(white_y_sorted)
        split_indices = np.where(gaps >= 34)[0]

        split_points = [0] + (split_indices + 1).tolist() + [len(white_y_sorted)]

        for i in range(len(split_points) - 1):
            sub_y = white_y_sorted[split_points[i]:split_points[i + 1]]
            if len(sub_y) == 0:
                continue

            y_top = int(np.min(sub_y))
            y_bottom = int(np.max(sub_y))

            # Setează pragurile minime pentru lățimea și înălțimea bounding box-ului (în pixeli)
            min_box_width = 28  # modifică după nevoie
            min_box_height = 50  # modifică după nevoie

            # Desenare
            cv2.line(result, (x, y_top), (x, y_bottom), (0, 0, 255), 2)
            cv2.rectangle(result, (left, y_top), (right, y_bottom), (255, 0, 0), 2)

            left_adj = max(left - 3, 0)
            right_adj = min(right + 3, w - 1)
            y_top_adj = max(y_top - 4, 0)
            y_bottom_adj = min(y_bottom + 4, h - 1)

            # Dimensiuni în pixeli
            bbox_width_pixels = right_adj - left_adj
            bbox_height_pixels = y_bottom_adj - y_top_adj

            # Dacă lățimea este sub prag, o mărim de 1.5 ori față de cea actuală (simetric)
            if bbox_width_pixels < (min_box_width+20):
                center_x = (left_adj + right_adj) // 2
                new_half_width = int((bbox_width_pixels * 1.28) / 2)

                left_adj = max(center_x - new_half_width, 0)
                right_adj = min(center_x + new_half_width, w - 1)
                bbox_width_pixels = right_adj - left_adj

                # Aplică doar dacă este suficient de lat și înalt
            if bbox_width_pixels >= min_box_width and bbox_height_pixels >= min_box_height:
                # Coordonate normalizate YOLO
                x_center = (left_adj + right_adj) / 2 / w
                y_center = (y_top_adj + y_bottom_adj) / 2 / h
                box_width = bbox_width_pixels / w
                box_height = bbox_height_pixels / h

                bboxes.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # Salvare imagine
    cv2.imwrite(os.path.join(final_output_folder, f"{name_only}.png"), result)

    # Salvare TXT cu aceleași nume
    with open(os.path.join(labels_output_folder, f"{name_only}.txt"), "w") as f:
        for bbox in bboxes:
            f.write(bbox + "\n")

    print(f"[✓] {filename} procesată și salvată cu TXT.")
end_total = time.time()
print(f"Timp: {end_total - start_total:.2f} secunde.")