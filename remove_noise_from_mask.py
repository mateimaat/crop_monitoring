import os
import cv2
import numpy as np

# Calea către măștile verzi generate anterior
input_folder = "/Users/mateimaat/Desktop/Metoda_HSV/green_masks"
output_folder = "/Users/mateimaat/Desktop/Metoda_HSV/green_masks_withot_noice"
min_area = 30

# Creează folderul de output dacă nu există
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png") and "green_mask_hsv" in filename:
        mask_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace("green_mask_hsv", "green_mask_noise_removed"))

        green_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if green_mask is None:
            print(f"Nu s-a putut încărca: {mask_path}")
            continue

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(green_mask, connectivity=8)
        filtered_mask = np.zeros_like(green_mask)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                filtered_mask[labels == i] = 255

        cv2.imwrite(output_path, filtered_mask)
        print(f"Salvat: {output_path}")