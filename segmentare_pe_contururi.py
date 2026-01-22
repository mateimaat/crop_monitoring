import cv2
import numpy as np

# Calea către imagine
image_path = '/Users/mateimaat/Desktop/data/sorghumfield.v3-416x416_augmented.yolov5pytorch/test/images/training_75_png.rf.23155c3496205e4094459310f857b293.jpg'  # <-- modifică aici cu calea reală
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Imaginea nu a fost găsită la: {image_path}")

# Conversie BGR -> HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Definirea pragurilor pentru verde
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Creăm masca pentru verde
mask = cv2.inRange(hsv, lower_green, upper_green)

# Definim elementul structurant (kernel) pentru dilatare
kernel = np.ones((5, 5), np.uint8)

# Aplicăm dilatarea
dilated_mask = cv2.dilate(mask, kernel, iterations=1)

# Afișăm și salvăm rezultatele
cv2.imwrite('masca_verde.jpg', mask)
cv2.imwrite('masca_verde_dilatata.jpg', dilated_mask)

cv2.imshow('Masca verde', mask)
cv2.imshow('Masca verde dupa dilatare', dilated_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()