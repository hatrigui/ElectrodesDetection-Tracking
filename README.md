# ElectrodesDetection-Tracking
## Project Description

This project aimed to develop a machine learning-based system for detecting and tracking electrodes in a plasma torch under extreme industrial conditions. The goal was to monitor electrode position and behavior in real-time, ensuring efficient and safe operation of plasma-based systems.

## Methodology

### Problem Context:
- **Plasma Torch**: Operates at temperatures exceeding 2,000°C, producing hydrogen (H₂) and carbon debris.
- **Challenges**: 
  - Electrode erosion that alters shape and size.
  - Light intensity fluctuations due to plasma.
  - Real-time tracking of three electrodes with occlusions and environmental noise.

### Data Preparation:
- **Video to Frames**: 7 videos split into 13,613 frames (70% training, 20% validation, 10% test).
- **Preprocessing**:
  - **Normal Conditions**: Gaussian blur, Otsu thresholding, contour detection, bounding boxes, YOLO-format normalization.
  - **Plasma Conditions**: Negative transformation, contrast adjustment, Gaussian blur, thresholding.

### Model Architecture:
- **YOLOv8** for electrode detection, combined with **Kalman Filters** for tracking.
- **DeepSORT** (CNN + Kalman Filter) for multi-object tracking (MOT).
- **TLD Framework** (Tracking-Learning-Detection) for real-time updates.

### Training & Evaluation:
- **Metrics**: Precision, recall, F1-score, confusion matrix, loss curves.

## Key Results

### Detection Performance:
- **YOLOv8 + Kalman Filter**:
  - Achieved high precision (~1.0 at confidence threshold 1.0) and F1-score (~0.74).
  - Accurately localized electrodes in most frames with minimal false positives.
  
### Challenges:
- **Partial Occlusions**: Missed detections when electrodes overlapped or were obscured.
- **Noise Sensitivity**: Fluctuating light intensity and heat artifacts caused false positives.
- **Electrode Erosion**: Shape changes reduced detection consistency over time.

### Proposed Improvements:
- **Kalman Filter Integration**: Stabilized bounding box predictions by leveraging motion history.
- **DeepSORT**: Enhanced multi-electrode tracking using re-identification features.

## Conclusion

The **YOLOv8** model combined with **Kalman Filters** demonstrated robust electrode detection in plasma torch environments, achieving high precision. However, challenges like occlusions and environmental noise require further optimization. Future work includes refining **DeepSORT** for better occlusion handling and integrating real-time feedback for industrial deployment.
