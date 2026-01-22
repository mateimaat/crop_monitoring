import cv2
import numpy as np
import os
from scipy.signal import find_peaks
import time

# Foldere
input_folder = "/Users/mateimaat/Desktop/DEMO/images"
intermediate_mask_folder = "/Users/mateimaat/Desktop/DEMO/masti_de_verde"
noise_removed_folder = "/Users/mateimaat/Desktop/DEMO/masti_fara_zgomot"
isolated_removed_folder = "/Users/mateimaat/Desktop/DEMO/masti_fara_izolate"
final_output_folder = "/Users/mateimaat/Desktop/DEMO/masti_cu_boxur"
labels_output_folder = "/Users/mateimaat/Desktop/DEMO/labels_pipeline"

os.makedirs(intermediate_mask_folder, exist_ok=True)
os.makedirs(noise_removed_folder, exist_ok=True)
os.makedirs(isolated_removed_folder, exist_ok=True)
os.makedirs(final_output_folder, exist_ok=True)
os.makedirs(labels_output_folder, exist_ok=True)

# Parametri
min_area = 5
start_total = time.time()
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    name_only = os.path.splitext(filename)[0]
    image_path = os.path.join(input_folder, filename)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Nu s-a putut încărca imaginea: {filename}")
        continue

    h, w = img.shape[:2]

    # PAS 1 - Detectare verde în HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    cv2.imwrite(os.path.join(intermediate_mask_folder, f"{name_only}.png"), green_mask)

    # PAS 1 - Detectare verde cu regula G - R > prag
    # b, g, r = cv2.split(img)
     #prag = 1
    #green_mask = np.zeros_like(g, dtype=np.uint8)
    #green_mask[(g.astype(int) - r.astype(int)) > prag] = 255
    #cv2.imwrite(os.path.join(intermediate_mask_folder, f"{name_only}.png"), green_mask)

    # PAS 2 - Eliminare zgomot
    grup_numar, grup, dim, _ = cv2.connectedComponentsWithStats(green_mask, connectivity=8)
    filtered_mask = np.zeros_like(green_mask)
    for i in range(1, grup_numar):
        if dim[i, cv2.CC_STAT_AREA] > min_area:
            filtered_mask[grup == i] = 255
    cv2.imwrite(os.path.join(noise_removed_folder, f"{name_only}.png"), filtered_mask)

    # PAS 3 - Eliminare plante izolate
    cleaned_mask = filtered_mask.copy()
    for x in range(w):
        coloana = filtered_mask[:, x]
        _, _, stats_coloana, _ = cv2.connectedComponentsWithStats(coloana.reshape(-1, 1), connectivity=8)
        if len(stats_coloana) - 1 < 2:
            cleaned_mask[:, x] = 0
    cv2.imwrite(os.path.join(isolated_removed_folder, f"{name_only}.png"), cleaned_mask)

    # PAS 4 - Proiecție verticală + detectare vârfuri
    vertical_projection = np.sum(cleaned_mask == 255, axis=0)
    peaks, _ = find_peaks(vertical_projection, distance=40, prominence=15)
    if vertical_projection[0] > 100:
        peaks = np.insert(peaks, 0, 0)
    if vertical_projection[-1] > 100:
        peaks = np.append(peaks, len(vertical_projection) - 1)

    masca_finala = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
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

            # Setează pragurile minime pentru lățimea și înălțimea bounding box-ului (în pixeli)
            min_latime = 28  # modifică după nevoie
            min_inaltime = 50  # modific

            stanga_3px = max(stanga - 3, 0)
            dreapta_3px = min(dreapta + 3, w - 1)
            rand_top_3px = max(rand_top - 3, 0)
            rand_low_3px = min(rand_low + 3, h - 1)

            # Dimensiuni în pixeli
            latime_box = dreapta_3px - stanga_3px
            inaltime_box = rand_low_3px - rand_top_3px

            # Dacă lățimea este sub prag, o mărim de 1.5 ori față de cea actuală (simetric)
            if latime_box < (min_latime + 8):
                center_x = (stanga_3px + dreapta_3px) // 2
                latime_jum = int((latime_box * 1.5) / 2)

                stanga_3px = max(center_x - latime_jum, 0)
                dreapta_3px = min(center_x + latime_jum, w - 1)
                latime_box = dreapta_3px - stanga_3px

            # Desenare
            cv2.line(masca_finala, (x, rand_top), (x, rand_low), (0, 0, 255), 2)
            cv2.rectangle(masca_finala, (stanga, rand_top), (dreapta, rand_low), (255, 0, 0), 2)

            # Aplică doar dacă este suficient de lat și înalt
            if latime_box >= min_latime and inaltime_box >= min_inaltime:
                # Coordonate normalizate YOLO
                x_center = (stanga_3px + dreapta_3px) / 2 / w
                y_center = (rand_top_3px + rand_low_3px) / 2 / h
                box_width = latime_box / w
                box_height = inaltime_box / h

                bboxes.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # Salvare imagine
    cv2.imwrite(os.path.join(final_output_folder, f"{name_only}.png"), masca_finala)

    # Salvare TXT cu aceleași nume
    with open(os.path.join(labels_output_folder, f"{name_only}.txt"), "w") as f:
        for bbox in bboxes:
            f.write(bbox + "\n")
    print(f"[✓] {filename} procesată și salvată cu TXT.")

end_total = time.time()
print(f"Timp: {end_total - start_total:.2f} secunde.")