import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# Load mask
mask_path = "/Users/mateimaat/Desktop/Green Masks/green_mask_noise_removed15.png"
green_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
output = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)

# Detect white pixels (where plants are)
coords = np.column_stack(np.where(green_mask == 255))

# Cluster white pixels based on x positions
db = DBSCAN(eps=15, min_samples=10).fit(coords[:, 1].reshape(-1, 1))

# Minimum height of a valid crop row (in pixels)
min_height = 100
# Minimum number of points in a cluster to be considered valid
min_points = 100

# Loop through each cluster (each crop row)
for label in set(db.labels_):
    if label == -1:
        continue  # skip noise

    group = coords[db.labels_ == label]

    # Skip small clusters
    if len(group) < min_points:
        continue

    x_mean = int(np.mean(group[:, 1]))
    y_min = int(np.min(group[:, 0]))
    y_max = int(np.max(group[:, 0]))

    # Skip clusters that are too short vertically
    if (y_max - y_min) < min_height:
        continue

    # Draw vertical line representing the crop row center
    cv2.line(output, (x_mean, y_min), (x_mean, y_max), (0, 0, 255), 2)

# Show result
cv2.imshow("Mijloc rand", output)
cv2.waitKey(0)
cv2.destroyAllWindows()