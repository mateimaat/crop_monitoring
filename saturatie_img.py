import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Calea către imaginea binară (mască de verde)
image_path = '/Users/mateimaat/Desktop/Metoda_HSV/green_masks_isolated/cleaned_green_mask_testing_28_png.rf.9749fcf39aea7932df9a36f8645b7381.png'  # înlocuiește cu calea ta

# Încarcă imaginea în tonuri de gri
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Verificare dacă imaginea a fost încărcată
if img is None:
    raise ValueError("Imaginea nu a fost găsită. Verifică calea.")

# Calculează proiecția verticală (suma pixelilor 255 pe fiecare coloană)
vertical_projection = np.sum(img == 255, axis=0)

# Creează folder de ieșire dacă nu există
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Afișare și salvare histogramă
plt.figure(figsize=(10, 4))
plt.plot(vertical_projection, color='green')
plt.title('Proiecție verticală')
plt.ylabel('Număr pixeli verzi')
plt.grid(True)
plt.tight_layout()

# Salvează figura în fișier PNG
output_path = os.path.join(output_folder, 'proiectie_verticala.png')
plt.savefig(output_path)
print(f"Graficul a fost salvat în: {output_path}")