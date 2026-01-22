import cv2
import numpy as np
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scipy.signal import find_peaks

def generate_labels_from_images(image_dir, labels_output_folder):
    os.makedirs(labels_output_folder, exist_ok=True)
    min_area = 5
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith((".jpg", ".png")):
            continue

        name_only = os.path.splitext(filename)[0]
        image_path = os.path.join(image_dir, filename)
        img = cv2.imread(image_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        prag = 1
        # HSV + G-R verde
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        #b, g, r = cv2.split(img)
        #green_mask = np.zeros_like(g, dtype=np.uint8)
        #green_mask[(g.astype(int) - r.astype(int)) > prag] = 255

        grup_numar, grup, dim, _ = cv2.connectedComponentsWithStats(green_mask, connectivity=8)
        filtered_mask = np.zeros_like(green_mask)
        for i in range(1, grup_numar):
            if dim[i, cv2.CC_STAT_AREA] > min_area:
                filtered_mask[grup == i] = 255

        cleaned_mask = filtered_mask.copy()
        for x in range(w):
            coloana = filtered_mask[:, x]
            _, _, stats_coloana, _ = cv2.connectedComponentsWithStats(coloana.reshape(-1, 1), connectivity=8)
            if len(stats_coloana) - 1 < 2:
                cleaned_mask[:, x] = 0

        vertical_projection = np.sum(cleaned_mask == 255, axis=0)
        peaks, _ = find_peaks(vertical_projection, distance=40, prominence=15)
        if vertical_projection[0] > 100:
            peaks = np.insert(peaks, 0, 0)
        if vertical_projection[-1] > 100:
            peaks = np.append(peaks, len(vertical_projection) - 1)

        bboxes = []
        for x in peaks:
            max_width = 55
            stanga = x
            dreapta = x
            for dx in range(1, max_width):
                if x - dx >= 0 and np.any(cleaned_mask[:, x - dx] == 255):
                    stanga = x - dx
                else:
                    break
            for dx in range(1, max_width):
                if x + dx < w and np.any(cleaned_mask[:, x + dx] == 255):
                    dreapta = x + dx
                else:
                    break

            randplante = cleaned_mask[:, stanga:dreapta + 1]
            white_coords = np.column_stack(np.where(randplante == 255))
            if len(white_coords) == 0:
                continue

            white_y = white_coords[:, 0]
            white_y_sorted = sorted(white_y)
            goluri = np.diff(white_y_sorted)
            split_indices = np.where(goluri >= 36)[0]
            pozitii_goluri = [0] + (split_indices + 1).tolist() + [len(white_y_sorted)]

            for i in range(len(pozitii_goluri) - 1):
                subrand = white_y_sorted[pozitii_goluri[i]:pozitii_goluri[i + 1]]
                if len(subrand) == 0:
                    continue
                rand_top = int(np.min(subrand))
                rand_low = int(np.max(subrand))

                stanga_3px = max(stanga - 3, 0)
                dreapta_3px = min(dreapta + 3, w - 1)
                rand_top_3px = max(rand_top - 3, 0)
                rand_low_3px = min(rand_low + 3, h - 1)

                latime_box = dreapta_3px - stanga_3px
                inaltime_box = rand_low_3px - rand_top_3px

                if latime_box < 36:
                    center_x = (stanga_3px + dreapta_3px) // 2
                    latime_jum = int((latime_box * 1.5) / 2)
                    stanga_3px = max(center_x - latime_jum, 0)
                    dreapta_3px = min(center_x + latime_jum, w - 1)
                    latime_box = dreapta_3px - stanga_3px

                if latime_box >= 28 and inaltime_box >= 50:
                    x_center = (stanga_3px + dreapta_3px) / 2 / w
                    y_center = (rand_top_3px + rand_low_3px) / 2 / h
                    box_width = latime_box / w
                    box_height = inaltime_box / h
                    bboxes.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

        label_path = os.path.join(labels_output_folder, f"{name_only}.txt")
        with open(label_path, "w") as f:
            for bbox in bboxes:
                f.write(bbox + "\n")

def calculate_green_coverage_with_bboxes(image_path, bbox_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    try:
        with open(bbox_path, "r") as f:
            bboxes = [line.strip().split() for line in f.readlines()]
    except FileNotFoundError:
        print(f"[WARNING] Lipsă fișier de etichete pentru: {image_path}")
        return None

    total_green_pixels = 0
    total_pixels = 0

    for bbox in bboxes:
        class_id, x_center, y_center, width, height = map(float, bbox)
        h, w = img.shape[:2]
        x_center, y_center = int(x_center * w), int(y_center * h)
        width, height = int(width * w), int(height * h)
        x1 = max(0, int(x_center - width / 2))
        y1 = max(0, int(y_center - height / 2))
        x2 = min(w, int(x_center + width / 2))
        y2 = min(h, int(y_center + height / 2))
        roi = hsv[y1:y2, x1:x2]
        mask = cv2.inRange(roi, lower_green, upper_green)
        green_pixels = cv2.countNonZero(mask)
        roi_pixels = roi.shape[0] * roi.shape[1]
        total_green_pixels += green_pixels
        total_pixels += roi_pixels

    return (total_green_pixels / total_pixels) * 100 if total_pixels > 0 else 0

def run_analysis(image_dir, text_output):
    labels_output_folder = os.path.join(image_dir, "labels_pipeline")
    generate_labels_from_images(image_dir, labels_output_folder)
    start_total = time.time()
    results = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(image_dir, filename)
            label_name = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(labels_output_folder, label_name)
            coverage = calculate_green_coverage_with_bboxes(image_path, label_path)
            if coverage is not None:
                results.append((filename, coverage))

    results.sort(key=lambda x: x[1])
    for filename, coverage in results:
        text_output.insert(tk.END, f"{filename} -> Grad acoperire verde: {coverage:.2f}%\n")

    end_total = time.time()
    text_output.insert(tk.END, f"\nTimp total: {end_total - start_total:.2f} secunde.\n")

def browse_directory(entry):
    path = filedialog.askdirectory()
    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)

def create_gui():
    root = tk.Tk()
    root.title("Detectie evolutie")
    tk.Label(root, text="Path imagini:").grid(row=0, column=0, sticky="e")
    image_entry = tk.Entry(root, width=60)
    image_entry.grid(row=0, column=1)
    tk.Button(root, text="Browse", command=lambda: browse_directory(image_entry)).grid(row=0, column=2)
    text_output = tk.Text(root, width=80, height=20)
    text_output.grid(row=2, column=0, columnspan=3, padx=10, pady=10)
    tk.Button(root, text="RUN", command=lambda: run_analysis(image_entry.get(), text_output)).grid(row=1, column=1, pady=10)
    root.mainloop()

if __name__ == "__main__":
    create_gui()
