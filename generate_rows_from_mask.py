import cv2
import numpy as np

mask_path = "/Users/mateimaat/Desktop/Green Masks/cleaned_green_mask15.png"
green_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

crop_rows_visual = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)

contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

grouped_rows = []
current_group = [bounding_boxes[0]]

for i in range(1, len(bounding_boxes)):
    x, y, w, h = bounding_boxes[i]
    prev_x, prev_y, prev_w, prev_h = bounding_boxes[i - 1]

    if x - (prev_x + prev_w) < 30:
        current_group.append(bounding_boxes[i])
    else:
        grouped_rows.append(current_group)
        current_group = [bounding_boxes[i]]

grouped_rows.append(current_group)

for group in grouped_rows:
    x_coords = [b[0] for b in group]
    y_coords = [b[1] for b in group]
    w_coords = [b[0] + b[2] for b in group]
    h_coords = [b[1] + b[3] for b in group]

    x_min, y_min = min(x_coords), min(y_coords)
    x_max, y_max = max(w_coords), max(h_coords)

    cv2.rectangle(crop_rows_visual, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue for full crop row

cv2.imshow("Crop Rows", crop_rows_visual)
cv2.waitKey(0)
cv2.destroyAllWindows()