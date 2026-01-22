import cv2
import numpy as np
from scipy.signal import find_peaks

# Load the binary green mask
mask_path = "/Users/mateimaat/Desktop/Metoda_HSV/green_masks_isolated/cleaned_green_mask_training_18_png.rf.945f918b7846052452e031aec2028822.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Step 1: Vertical projection (suma pixelilor albi pe fiecare coloană)
vertical_projection = np.sum(mask == 255, axis=0)

# Step 2: Detectare vârfuri (coloane unde sunt rânduri de plante)
peaks, _ = find_peaks(vertical_projection, distance=30, prominence=80)

# Step 3: Adăugare margini dacă sunt parțial detectate
threshold = 5
if vertical_projection[0] > threshold:
    peaks = np.insert(peaks, 0, 0)
if vertical_projection[-1] > threshold:
    peaks = np.append(peaks, len(vertical_projection) - 1)

# Step 4: Procesare fiecare linie de mijloc
for x in peaks:
    max_width = 100
    left = x
    right = x

    # Căutăm stânga
    for dx in range(1, max_width):
        if x - dx >= 0 and np.any(mask[:, x - dx] == 255):
            left = x - dx
        else:
            break

    # Căutăm dreapta
    for dx in range(1, max_width):
        if x + dx < mask.shape[1] and np.any(mask[:, x + dx] == 255):
            right = x + dx
        else:
            break

    # Selectăm doar zona dintre left și right
    subregion = mask[:, left:right + 1]
    white_coords = np.column_stack(np.where(subregion == 255))

    if len(white_coords) == 0:
        continue

    y_top = int(np.min(white_coords[:, 0]))
    y_bottom = int(np.max(white_coords[:, 0]))

    # Linie de mijloc
    cv2.line(output, (x, y_top), (x, y_bottom), (0, 0, 255), 2)

    # Bounding box complet
    cv2.rectangle(output, (left, y_top), (right, y_bottom), (255, 0, 0), 2)

# Afișare și salvare
cv2.imshow("Linii de centru detectate", output)
cv2.imwrite("/Users/mateimaat/Desktop/boxed_mask.png", output)
cv2.waitKey(0)
cv2.destroyAllWindows()