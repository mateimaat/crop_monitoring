import cv2
import numpy as np
import matplotlib.pyplot as plt

mask_path = "/Users/mateimaat/Desktop/Metoda_HSV/green_masks_isolated/cleaned_green_mask_training_122_png.rf.e0a857770d48399eb4e8ed75f0469b9e.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

projection = np.sum(mask == 255, axis=0)
window_size = 5
projection_smooth = np.convolve(projection, np.ones(window_size)/window_size, mode='same')

plt.figure(figsize=(12, 4))
plt.plot(projection_smooth, label="Smoothed Projection", color="orange")
plt.title("Proiectie pe orizontala")
plt.ylabel("Nr.pixeli")
plt.grid(True)
plt.tight_layout()

output_path = "/Users/mateimaat/Desktop/histograma_out111.png"
plt.savefig(output_path)