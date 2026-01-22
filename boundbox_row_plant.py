import os


def combine_detections(rows_folder, plants_folder, output_folder):
    # Creează folderul de output dacă nu există
    os.makedirs(output_folder, exist_ok=True)

    # Lista fișierelor din folderul de rânduri
    rows_files = os.listdir(rows_folder)

    for file in rows_files:
        # Verifică dacă există un fișier corespunzător în folderul de plante
        rows_file_path = os.path.join(rows_folder, file)
        plants_file_path = os.path.join(plants_folder, file)

        # Cale pentru fișierul de output
        output_file_path = os.path.join(output_folder, file)

        # Citire detectări rânduri
        with open(rows_file_path, 'r') as rf:
            rows_content = rf.readlines()

        # Citire detectări plante
        plants_content = []
        if os.path.exists(plants_file_path):
            with open(plants_file_path, 'r') as pf:
                plants_content = pf.readlines()

        # Combina rândurile și plantele
        combined_content = rows_content + plants_content

        # Scrie fișierul combinat în folderul de output
        with open(output_file_path, 'w') as of:
            of.writelines(combined_content)

    print(f"Fișierele combinate au fost salvate în: {output_folder}")


rows_folder = "/Users/mateimaat/Desktop/yolo_results/exp3/labels"  # Folderul cu detectări pentru rânduri
plants_folder = "/Users/mateimaat/Desktop/yolo_results/boundboxes_plants"  # Folderul cu detectări pentru plante
output_folder = "/Users/mateimaat/Desktop/yolo_results/bound_plant_row"  # Folderul pentru output

combine_detections(rows_folder, plants_folder, output_folder)