## 🌪️ Typhoon Detection in Satellite Images using YOLOv5

## 📌 Project Overview
This project focuses on object detection of typhoons in satellite imagery using deep learning. The goal is to automatically identify and localize typhoon formations from satellite images to support environmental monitoring and disaster analysis.

## Objectives
* Detect typhoons from satellite images using deep learning
* Train and fine-tune a YOLOv5-based object detection model
* Improve model generalization using data preprocessing and augmentation
* Evaluate model performance using standard object detection metrics

## 🧠 Model & Approach
* Model Used: YOLOv5
* Framework: PyTorch
* Approach:
    * Satellite images are preprocessed and augmented
    * YOLOv5 is trained to detect typhoon regions via bounding boxes
    * Model predictions are evaluated using standard metrics

## 📂 Dataset
* Type: Satellite images containing typhoon patterns
* Size: 10,000+ images
* Preprocessing & Augmentation:
    * Resizing and normalization
    * Image augmentation to improve robustness and generalization
## 🛠️ Tech Stack
* Programming Language: Python
* Deep Learning Framework: PyTorch
* Model: YOLOv5
* Libraries & Tools:
    * OpenCV
    * NumPy
    * Matplotlib (for visualization)
## 📤 Output
* Bounding boxes highlighting detected typhoon regions in satellite images
* Confidence scores for detected objects
* Quantitative performance metrics (mAP, F1-score, precision, recall)

## 🚀 Results
* Achieved over 90% detection accuracy
* Improved generalization through data augmentation
* Demonstrated effective use of YOLOv5 for environmental object detection tasks
