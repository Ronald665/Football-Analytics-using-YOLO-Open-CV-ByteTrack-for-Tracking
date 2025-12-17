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
NB: dataset can be found in training Folder<br>

## 4. Methodology:
### 4.1 Model Selection and Custom Training
Initial inference was performed using a pre-trained YOLOv5 model; however, this resulted in frequent false detections of spectators and non-relevant individuals outside the pitch. 
<p align="center">
  <img width="770" height="430" alt="Screenshot 2025-12-17 213015" src="https://github.com/user-attachments/assets/d570690a-f796-4d73-ba34-bae6b1965044" />
</p>
<p align="center">
  <em><strong>Figure 1:</strong> YOLOv5 Video Inference Prior to Fine-Tuning.</em>
</p>

**Link to video:** https://www.dropbox.com/scl/fi/0oj0frrfl2fy32ankg2ha/08fd33_4.avi?rlkey=4xy5aerikqs1kw15wjybe0pr3&dl=0 <br>

To address this limitation, the model was fine-tuned on a custom football-specific dataset obtained from Roboflow, containing annotated examples of players, referees, and the ball.\

<p align="center">
<img width="770" height="430" alt="video after training" src="https://github.com/user-attachments/assets/3b2b0754-2524-4b2a-8909-cf3fe73e4bc6" />
</p>
<p align="center">
  <em><strong>Figure 1:</strong> Video after Training on custom dataset.</em>
</p>

**Link to video:** [https://www.dropbox.com/scl/fi/0oj0frrfl2fy32ankg2ha/08fd33_4.avi?rlkey=4xy5aerikqs1kw15wjybe0pr3&dl=0](https://www.dropbox.com/scl/fi/r2qcdn874vuu401aq2tlv/after.avi?rlkey=e5acqztv9f7mjq34px1tpl6w8&dl=0) 
<br>

NB: Actual training was done using google collab

**Result OF training on Custom Dataset**
| Class        | Images | Instances | Precision (P) | Recall (R) | mAP@50 | mAP@50–95 |
|-------------|--------|-----------|---------------|------------|--------|-----------|
| Ball        | 35     | 35        | 0.226         | 0.386      | 0.170  | –         |
| Goalkeeper | 27     | 27        | 0.917         | 0.926      | 0.973  | 0.744     |
| Player     | 38     | 754       | 0.974         | 0.959      | 0.987  | 0.800     |
| Referee    | 38     | 89        | 0.941         | 0.889      | 0.935  | 0.648     |
| **All**    | 38     | 905       | 0.958         | 0.750      | 0.820  | 0.591     |


* Average processing speed per image: 9.1 ms inference, 2.6 ms post-processing.
* Player and referee detection performed very well (mAP50 > 0.93)
* Ball detection had lower performance (mAP50 = 0.17), consistent with its small size and rapid movement.

Despite the lower ball detection metrics, the system was able to track ball movement and player possession effectively through temporal interpolation and association with nearest players. Player tracking and team classification remained highly accurate for most frames, with minor misassignments towards the end of the video.

The trained model was subsequently adopted as the base detector for all downstream prediction and analysis tasks, significantly improving on-pitch detection accuracy and reducing background false positives.
<br><br>
NB: this model can be found in 'model' under the name 'best.pt' <br> <br>

### 4.2 Object Detection and Tracking Methodology
Object detection was performed using a YOLO-based deep learning model to identify players, referees, and the ball in each video frame. To ensure computational efficiency, frames were processed in batches during inference. Detected objects were subsequently converted into a unified detection format and passed to the ByteTrack multi-object tracking algorithm to maintain consistent identities across frames.

To simplify analysis, goalkeeper detections were merged with player detections, allowing all on-field players to be treated as a single class during tracking. Player and referee objects were assigned persistent track IDs, while the ball—being a fast-moving and frequently occluded object—was handled separately without long-term ID association.
Missed ball detections were addressed through temporal interpolation of bounding box coordinates using linear interpolation, ensuring a continuous ball trajectory across frames. This improved robustness for downstream tasks such as possession estimation and ball movement analysis.

### 4.3 Visualization and Analytics
Custom visualization techniques were applied to enhance interpretability. Elliptical markers were drawn at the base of player and referee bounding boxes to approximate ground contact, while triangular markers were used to indicate ball location and player possession. Team ball possession statistics were computed cumulatively across frames and displayed as a semi-transparent overlay on the video output.

### 4.4 Team Classification
Team identification was performed using an unsupervised color-based clustering approach. For each detected player, the bounding box region was extracted from the video frame, and only the upper half of the bounding box was used to minimize background and pitch interference. Pixel values from this region were reshaped into a two-dimensional array and clustered using K-Means with two clusters, representing foreground (jersey) and background colors.

To isolate the player’s jersey color, corner pixels of the cropped region were assumed to belong to the background cluster, while the remaining cluster was selected as the player cluster. The corresponding cluster centroid was used as the representative color of the player.

Once representative colors were extracted for all players in a frame, a second K-Means clustering step was applied to group players into two teams based on color similarity. These cluster centroids were stored as team color references. Each player was subsequently assigned a team label by comparing their extracted color to the learned team centroids. Player-to-team mappings were cached to ensure temporal consistency across frames.

### 4.5 Player–Ball Association
To determine player possession, a distance-based player–ball association method was employed. For each frame, the ball position was estimated using the center of its detected bounding box. The Euclidean distance between the ball and each detected player was then computed using the lower corners of the player’s bounding box, approximating foot location where ball interaction typically occurs.

The player with the minimum distance to the ball, provided the distance was below a predefined threshold, was assigned possession of the ball. This threshold-based filtering reduced false assignments when the ball was not in close proximity to any player. The approach enabled reliable estimation of individual possession events, which were subsequently aggregated for team-level possession analysis.

## 5 Results
The custom-trained YOLOv5 model was able to detect players, referees, and the ball on the pitch and reduced false detections outside the playing area compared to the pre-trained model. Player tracking worked well for most of the video, but toward the end of the sequence, one or two players were incorrectly assigned to the wrong team.

<p align="center">
  <img width="767" height="425" alt="Screenshot 2025-12-17 220150" src="https://github.com/user-attachments/assets/9c88dbf2-88b5-4b23-8332-50243bdbb612" />
</p>
<p align="center">
  <em><strong>Figure 1:</strong> Final Output.</em>
</p>

**Link to video: ** https://www.dropbox.com/scl/fi/znzwobl80j8x6kuwh8nx2/output_video.avi?rlkey=ba16ogbsvl6ro3jp3mxkmi2lr&dl=0 <br>

The jersey color–based team assignment was mostly accurate, although a small number of misclassifications occurred due to changes in appearance and tracking inconsistencies. Ball tracking was generally reliable during short passes and close play; however, during long-ball situations, the tracker sometimes drifted away from the actual ball position over a long distance.

Despite these limitations, the system produced reasonable team possession estimates and demonstrated the feasibility of automated football video analysis using deep learning and computer vision techniques.

## Future Work
Future improvements could include upgrading the detection backbone to more recent architectures such as YOLOv8 or YOLOv11, as well as transformer-based object detection models like DETR or Meta AI’s DINO / Grounding DINO, which may improve robustness to occlusion and long-range object tracking. Incorporating explicit camera motion compensation would enable more accurate estimation of player displacement, allowing reliable computation of player speed and movement-based metrics.

Additional extensions could involve extracting higher-level football statistics such as distance covered, sprint frequency, pressing intensity, and spatial heatmaps. Improving ball tracking during long passes using motion models such as Kalman filtering could further enhance possession and event analysis.
