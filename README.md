# Football Analytics using YOLO, Open CV,  Bytetrack for Object Tracking

## 1 AIM:
This project presents an automated football analytics system using deep learning and computer vision. YOLO a deep learning object detector model based on the CNN architecture 
was trained on a custom dataset to identify players, referees, and the ball from match footage. Object tracking, jersey-based team classification, ball localization and  
possession analysis were implemented. 

## 2 Introduction:
Recent advances in deep learning and computer vision have enabled automated analysis of sports videos, allowing meaningful information to be extracted directly from match 
footage. In particular, convolutional neural network (CNN)-based object detection models have shown strong performance in complex visual environments.This project presents 
an automated football analytics system using deep learning and computer vision. A YOLO-based object detector was trained on a custom dataset to identify players, referees, and 
the ball from football match videos. In addition to object detection, the system performs object tracking, jersey-based team classification, ball localization, and team
possession analysis. By integrating these components, the proposed approach demonstrates how raw football video can be transformed into structured and quantitative analytical
insights.

## 3 Dataset and Preprocessing
### 3.1 Video Data
The video data used in this project consists of a 31-second broadcast football match clip from the German Bundesliga, featuring two professional teams competing on a standard
football pitch. The video was captured from a broadcast camera perspective, which includes camera motion, zoom variations, and dynamic scene changes typical of real-world
football footage. This clip was used to evaluate the performance of the trained model on realistic match conditions.
### 3.2 Training Dataset
The object detection model was trained on a publicly available annotated dataset from Roboflow, containing 663 images with four classes: players, goalkeepers, referees, and the
ball. Football players were annotated in YOLOv5 PyTorch format.

To augment the dataset, Roboflow applied preprocessing and augmentation to each image, creating three versions per source image with:
* 50% probability of horizontal flip
* Random brightness adjustment between -20% and +20%
