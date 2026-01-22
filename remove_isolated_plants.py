import os
import cv2
import numpy as np

# Foldere de intrare și ieșire
input_folder = "/Users/mateimaat/Desktop/Metoda_HSV/green_masks_withot_noice"
output_folder = "/Users/mateimaat/Desktop/Metoda_HSV/green_masks_isolated"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png") and "green_mask_noise_removed" in filename:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace("green_mask_noise_removed", "cleaned_green_mask"))

        green_mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if green_mask is None:
            print(f"Nu s-a putut citi: {input_path}")
            continue

        filtered_mask = green_mask.copy()
        height, width = green_mask.shape

        for x in range(width):
            column = green_mask[:, x]
            _, labels, stats, _ = cv2.connectedComponentsWithStats(column.reshape(-1, 1), connectivity=8)
            num_groups = len(stats) - 1

            if num_groups < 2:
                filtered_mask[:, x] = 0  # Șterge coloana dacă are doar o grupare de pixeli

        cv2.imwrite(output_path, filtered_mask)
        print(f"Salvat: {output_path}")