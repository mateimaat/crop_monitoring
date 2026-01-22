import os
import csv

def parse_txt(file_path, target_class):
    """
    Reads a .txt file and extracts bounding boxes for the specified target class.
    """
    boxes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                if int(data[0]) == target_class:  # Filter by class
                    boxes.append(data[1:])  # Keep only bounding box coordinates
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return boxes

def generate_summary_report(bound_row_path, coverage_data_folder, output_csv):
    """
    Generates a summary report by extracting row and plant counts, and green coverage values from the dataset.
    """
    print(f"Generating report for the entire folder...")

    # Create or overwrite the CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Rows", "Plants", "Total_green", "Box_green"])

        # Iterate through all files in the coverage folder
        for file_name in os.listdir(coverage_data_folder):
            if file_name.endswith(".txt"):
                print(f"Processing file: {file_name}")

                # Construct paths for associated text files
                bound_file = os.path.join(bound_row_path, file_name)
                coverage_file = os.path.join(coverage_data_folder, file_name)

                # Check if files exist
                if not os.path.exists(bound_file):
                    print(f"Missing bounding box file: {bound_file}")
                    continue
                if not os.path.exists(coverage_file):
                    print(f"Missing coverage file: {coverage_file}")
                    continue

                # Count rows and plants from the same file
                rows_count = len(parse_txt(bound_file, target_class=0))  # Class 0 for rows
                plants_count = len(parse_txt(bound_file, target_class=1))  # Class 1 for plants
                print(f"Detected {rows_count} rows and {plants_count} plants for {file_name}")

                # Read green coverage data
                green_total = 0
                green_in_boxes = 0
                try:
                    with open(coverage_file, 'r') as cf:
                        data = cf.readlines()
                        if len(data) >= 2:
                            green_total = float(data[0].strip().split(":")[1].replace("%", ""))
                            green_in_boxes = float(data[1].strip().split(":")[1].replace("%", ""))
                except Exception as e:
                    print(f"Error reading coverage file {coverage_file}: {e}")

                print(f"Green coverage for {file_name} - Total: {green_total}%, Bounding Boxes: {green_in_boxes}%")

                # Write data to CSV
                writer.writerow([file_name, rows_count, plants_count, green_total, green_in_boxes])

    print(f"Summary report for all images saved to: {output_csv}")

# Example usage
bound_txt_folder = "/Users/mateimaat/Desktop/yolo_results/bound_plant_row"  # Folder with both rows & plants
coverage_folder = "/Users/mateimaat/Desktop/yolo_results/coverage_data"  # Folder with green coverage data
output_csv_file = "/Users/mateimaat/Desktop/yolo_results/summary_report_all.csv"  # Output CSV file

generate_summary_report(bound_txt_folder, coverage_folder, output_csv_file)