import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os

# Folderul de intrare și ieșire
input_folder = "/Users/mateimaat/Desktop/Metoda_HSV/green_masks_isolated"
output_folder = "/Users/mateimaat/Desktop/Metoda_HSV/green_masks_rows"
os.makedirs(output_folder, exist_ok=True)

# Parametri
min_height = 50
min_points = 100
max_width = 100

# Parcurge toate imaginile din folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png") and "cleaned_green_mask" in filename:
        mask_path = os.path.join(input_folder, filename)
        green_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if green_mask is None:
            print(f"[!] Nu s-a putut încărca: {mask_path}")
            continue

        output = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
        coords = np.column_stack(np.where(green_mask == 255))

        if len(coords) == 0:
            print(f"[{filename}] - Fără pixeli albi detectați.")
            continue

        db = DBSCAN(eps=15, min_samples=10).fit(coords[:, 1].reshape(-1, 1))

        for label in set(db.labels_):
            if label == -1:
                continue

            group = coords[db.labels_ == label]
            if len(group) < min_points:
                continue

            x_center = int(np.mean(group[:, 1]))
            y_min = int(np.min(group[:, 0]))
            y_max = int(np.max(group[:, 0]))
            if (y_max - y_min) < min_height:
                continue

            # Linie centrală
            cv2.line(output, (x_center, y_min), (x_center, y_max), (0, 0, 255), 2)

            # Căutare laterală
            left_bound = x_center
            right_bound = x_center

            for dx in range(1, max_width):
                if x_center - dx >= 0 and np.any(green_mask[y_min:y_max, x_center - dx] == 255):
                    left_bound = x_center - dx
                else:
                    break

            for dx in range(1, max_width):
                if x_center + dx < green_mask.shape[1] and np.any(green_mask[y_min:y_max, x_center + dx] == 255):
                    right_bound = x_center + dx
                else:
                    break

            # Bounding box
            cv2.rectangle(output, (left_bound, y_min), (right_bound, y_max), (255, 0, 0), 2)

        # Salvează imaginea rezultată
        out_path = os.path.join(output_folder, filename.replace("cleaned_green_mask", "boxed"))
        cv2.imwrite(out_path, output)
        print(f"[✓] Salvat: {out_path}")