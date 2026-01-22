import cv2
import numpy as np
from scipy.signal import find_peaks
import os

# FOLDER INPUT și OUTPUT
input_folder = "/Users/mateimaat/Desktop/Metoda_HSV/green_masks_isolated"
output_folder = "/Users/mateimaat/Desktop/Metoda_HSV/green_masks_rows"
os.makedirs(output_folder, exist_ok=True)

# Parametri
max_width = 100
threshold = 5
peak_distance = 30
peak_prominence = 30

# Procesare fiecare imagine
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png") and "cleaned_green_mask" in filename:
        mask_path = os.path.join(input_folder, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[!] Nu s-a putut încărca: {mask_path}")
            continue

        output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Step 1: Vertical projection
        vertical_projection = np.sum(mask == 255, axis=0)

        # Step 2: Detectare vârfuri
        peaks, _ = find_peaks(vertical_projection, distance=peak_distance, prominence=peak_prominence)

        # Step 3: Adăugare margini dacă sunt parțial detectate
        if vertical_projection[0] > threshold:
            peaks = np.insert(peaks, 0, 0)
        if vertical_projection[-1] > threshold:
            peaks = np.append(peaks, len(vertical_projection) - 1)

        # Step 4: Procesare fiecare linie de mijloc
        for x in peaks:
            left = x
            right = x

            # Căutare stânga
            for dx in range(1, max_width):
                if x - dx >= 0 and np.any(mask[:, x - dx] == 255):
                    left = x - dx
                else:
                    break

            # Căutare dreapta
            for dx in range(1, max_width):
                if x + dx < mask.shape[1] and np.any(mask[:, x + dx] == 255):
                    right = x + dx
                else:
                    break

            # Subregiune
            subregion = mask[:, left:right + 1]
            white_coords = np.column_stack(np.where(subregion == 255))

            if len(white_coords) == 0:
                continue

            y_top = int(np.min(white_coords[:, 0]))
            y_bottom = int(np.max(white_coords[:, 0]))

            # Linie de mijloc
            cv2.line(output, (x, y_top), (x, y_bottom), (0, 0, 255), 2)

            # Bounding box
            cv2.rectangle(output, (left, y_top), (right, y_bottom), (255, 0, 0), 2)

        # Salvare imagine rezultată
        output_path = os.path.join(output_folder, filename.replace("cleaned_green_mask", "boxed"))
        cv2.imwrite(output_path, output)
        print(f"[✓] Salvat: {output_path}")