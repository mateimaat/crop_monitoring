import matplotlib
matplotlib.use('Agg')  # Use a backend suitable for environments without a display server
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_path = "/Users/mateimaat/Desktop/yolo_results/summary_report_all.csv"
data = pd.read_csv(csv_path)

# Inspect the data
print(data.head())

# Visualization 1: Histogram of Rows and Plants
plt.figure(figsize=(10, 6))
plt.hist(data['Rows Detected'], bins=10, alpha=0.7, label='Rows Detected')
plt.hist(data['Plants Detected'], bins=10, alpha=0.7, label='Plants Detected')
plt.title('Distribution of Rows and Plants Detected')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('/Users/mateimaat/Desktop/yolo_results/histogram_rows_plants.png')  # Save to file
plt.close()

# Visualization 2: Bar Chart of Green Coverage
plt.figure(figsize=(12, 6))
plt.bar(data['Image'], data['Green Coverage Total (%)'], alpha=0.7, label='Total Green Coverage')
plt.bar(data['Image'], data['Green Coverage in Bounding Boxes (%)'], alpha=0.7, label='Green Coverage in Boxes')
plt.title('Green Coverage Comparison Across Images')
plt.xlabel('Image')
plt.ylabel('Green Coverage (%)')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/mateimaat/Desktop/yolo_results/bar_chart_coverage.png')  # Save to file
plt.close()

# Visualization 3: Scatter Plot of Rows vs. Plants Detected
plt.figure(figsize=(10, 6))
plt.scatter(data['Rows Detected'], data['Plants Detected'], c='blue', alpha=0.7)
plt.title('Relationship Between Rows and Plants Detected')
plt.xlabel('Rows Detected')
plt.ylabel('Plants Detected')
plt.grid(True)
plt.savefig('/Users/mateimaat/Desktop/yolo_results/scatter_rows_plants.png')  # Save to file
plt.close()