# Crop Evolution Stage Detection in Color Aerial Images

## Project Overview
This project was developed as a diploma thesis at the **National University of Science and Technology POLITEHNICA Bucharest**.  
It focuses on the automated analysis of agricultural crop development using color aerial imagery captured by UAV platforms.

The system combines traditional computer vision techniques with modern deep learning approaches to detect crop rows and estimate plant growth stages throughout the vegetation cycle.

---

## Features

### Automated Plant Row Detection
- Identification of geometric trajectories of crop rows from aerial images
- Robust detection under varying lighting and field conditions

### Dual-Method Approach

#### Color-Based Pipeline (Custom Method)
- HSV color space segmentation for vegetation extraction
- G–R (Green–Red) thresholding rules for fast crop isolation

#### Deep Learning Pipeline (YOLO)
- Integration of YOLOv5 and YOLOv8 architectures
- Nano and Small variants evaluated for accuracy–speed trade-offs
- Robust detection of crop rows under complex backgrounds

### Performance Evaluation
- Quantitative benchmarking using:
  - Intersection over Union (IoU)
  - Mean Average Precision (mAP)
  - Precision and Recall
- Comparative analysis between classical and deep learning methods

### Growth Stage Estimation
- Crop evolution assessment via green pixel density
- Percentage of vegetation pixels calculated inside detected bounding boxes
- Clear correlation between growth stages and vegetation coverage

---

## Technologies Used
- **Python** – Core programming language
- **OpenCV** – Image processing and color space transformations
- **PyTorch** – Training and inference for YOLO models
- **NumPy** – Numerical computations and array processing
- **SciPy** – Signal processing and histogram peak detection
- **Pandas** – Dataset handling, metric aggregation, and result analysis

---

## Methodology

### 1. Color-Based Method (Custom Pipeline)

#### Preprocessing
- Conversion from RGB to HSV color space
- Normalization of color channels

#### Segmentation
- Green vegetation extraction using HSV thresholds
- Application of G–R color index rules

#### Noise Removal
- Connected components analysis to remove isolated pixels
- Morphological operations for mask refinement

#### Row Detection
- Vertical projection of vegetation masks
- Histogram peak detection to identify row centerlines

#### Bounding Box Generation
- Automatic bounding box construction around dense vegetation areas
- Green pixel percentage computed per bounding box

---

### 2. Deep Learning Method (YOLO)

#### Models Used
- YOLOv5n
- YOLOv5s
- YOLOv8n
- YOLOv8s

#### Dataset
- Sorghum Crop Line Detection Dataset  
  (University of Purdue, Kaggle)

#### Training Strategy
- Data augmentation techniques:
  - Mosaic augmentation
  - Scaling
  - Color space manipulation
- Optimization and evaluation using standard detection metrics

---

## Key Results
- YOLOv8s achieved the highest overall performance with a mean IoU of **0.769**
- The color-based G–R method achieved competitive performance with an IoU of **0.746**
- The custom pipeline demonstrated significantly faster execution:
  - Color-based method: **0.54 seconds**
  - YOLO-based method: **~7 seconds**
- A consistent upward trend in green pixel percentage was observed as crops matured

---

## Project Structure

