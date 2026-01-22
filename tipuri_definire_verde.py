import cv2
import numpy as np

# Încarcă imaginea
image_path = "/Users/mateimaat/Desktop/data/sorghumfield.v3-416x416_augmented.yolov5pytorch/test/images/testing_17_png.rf.2cbfe9d703e1046a6bd252360814d983.jpg"  # 👉 înlocuiește cu calea ta
img = cv2.imread(image_path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# --- Masca G - R > prag ---
b, g, r = cv2.split(img)
prag = 2  # Poți modifica
green_mask_gr = np.zeros_like(g, dtype=np.uint8)
green_mask_gr[(g.astype(int) - r.astype(int)) > prag] = 255

# --- Masca HSV (Hue între 30 și 80) ---
lower_green = np.array([30, 40, 40])
upper_green = np.array([80, 255, 255])
green_mask_hsv = cv2.inRange(hsv, lower_green, upper_green)

# Afișare
cv2.imshow("Original", img)
cv2.imshow(f"Green Mask G - R > {prag}", green_mask_gr)
cv2.imshow("Green Mask HSV", green_mask_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()