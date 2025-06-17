# Project Description: Raspberry Pi Microscope with YOLOv8 Object Detection

This project implements a deep learning-based leaf object detection system using a Raspberry Pi 4 equipped with an HQ Camera. The system processes real-time microscope feed through a YOLOv8s model and displays detections on a 4.3" IPS LCD screen. The system consists of:
- Raspberry Pi 4 with HQ Camera
- Waveshare Microscope Kit (12MP visual magnification)
- 4.3" IPS LCD screen (800×480)
- Custom-trained YOLOv8s model for leaf object detection

 Created a specialized dataset by capturing images of 10 different leaf types

## Annotation
LabelImg is a free, open-source, graphical image annotation tool used for labeling objects within images, primarily for computer vision tasks like object detection.
- Tool: LabelImg 
- Format: YOLO text files
- Example annotation: 1 0.514670 0.477961 0.844428 0.440789
            (class_id x_center y_center width height)

## Dataset Preparation
The dataset creation process involved:
1. Capturing 219 high-resolution images (159 train, 30 validation, 30 test) using the Raspberry Pi microscope
2. Annotating each image with LabelImg, drawing precise bounding boxes around leaves
3. Storing annotations in YOLO format text files with normalized coordinates

The training pipeline uses a data.yml configuration file that species:
1. Paths to the training,validation,testing image directories 
2. Number of classes  # example nc= 10
3. Class names

## Classes
10 classes

["Ivy", "Fern", "Ginkgo", "Kummerowia striata", "Laciniata", "Macrolobium acaciifolium", "Micranthes odontoloma", "Murraya", "Robinia pseudoacacia", "Selaginella davidi franch"] 

## Model Training
Employed transfer learning with the YOLOv8s (small) pretrained model through Ultralytics Hub and the trained model was converted to ONNX format
- epochs: 10
- batch=5
- input size= 416

## Deployment
The trained ONNX model was deployed on Raspberry Pi microscope for real-time leaf detection on the Raspberry Pi, processing live feed and displaying bounding boxes with class labels. 
To deploy the trained ONNX model on the Raspberry Pi:
1. Activate the virtual environment by runnning:
```shell
source myenv/bin/activate
```
2. Install required dependencies with: 
```shell   
pip install -r requirements.txt
```     
3. Execute the ONNX model using onnxruntime by running:
```shell  
python3 YOLOv8s.py
```    
The script opens the camera then captures image and predicts the new image. 

          
         
         
         
         
         
         




