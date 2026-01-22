import cv2
import numpy as np

def calculate_green_coverage(image_path):
    # Citește imaginea
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definește intervalul de culoare pentru verde
    lower_green = np.array([35, 40, 40])  # Ajustează pentru specificul culturii tale
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Calculează procentul de pixeli verzi
    green_pixels = cv2.countNonZero(mask)
    total_pixels = img.shape[0] * img.shape[1]
    green_coverage = (green_pixels / total_pixels) * 100

    return green_coverage, mask

# Testează pe o imagine
image_path = "/Users/mateimaat/Desktop/yolo_results/exp3/testing_17_png.rf.2cbfe9d703e1046a6bd252360814d983.jpg"  # Înlocuiește cu calea unei imagini de test
coverage, mask = calculate_green_coverage(image_path)

# Afișează rezultatul
print(f"Gradul de acoperire verde: {coverage:.2f}%")

# Opțional: afișează masca
cv2.imshow("Masca Verde", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()