README

Dog Detection, Posture Classification & Pose Estimation (YOLOv8)

1. Overview
This project demonstrates a multi-stage computer vision pipeline using YOLOv8 for:
• Dog detection
• Posture classification (sitting vs standing)
• Pose/keypoint estimation (6 annotated attributes)
The goal was to progressively build more complex models starting from object detection to fine-grained pose estimation.


2. Approach
a. Dog Detection
• Started with a pretrained YOLOv8 model (trained on COCO)
• Fine-tuned using a custom dataset of dog images
• Goal: reliably detect dogs in diverse scenes

b. Posture Classification (Sitting vs Standing)
• Used the detection model as a base
• Manually labeled images into:
• sitting
• standing
• Fine-tuned YOLOv8 for classification
Result:
• Strong convergence
• High accuracy with minimal errors
• Clear separation between classes

c. Pose / Keypoint Detection
• Trained a YOLOv8 pose model
• Manually labeled 6 keypoints (dog body attributes)
Result:
• Predictions are less stable
• Keypoint placement is inconsistent compared to classification model

This is because I trained the model on 785 images and this size is not large enough for training Keypoint Reference.
Due to keypoints being a more complex task for the model rather than classification. This small dataset with 6 attributes leads to poor generalization.
In addition, the keypoint model showed weaker performance was also affected by the challenges in maintaining consistent keypoint placement across varied dog poses. I made sure to place keypoints on the same attribute, however, differences in dog poses contributed to inconsistencies. 


3. Demo Application
A Streamlit app was built to showcase results:
• Displays detection and pose outputs side-by-side
• Uses preselected demo images for immediate visualization
• Allows optional user image upload
Run locally:
streamlit run app.py


4. Project Structure
app/app.py                 
models/                
demos/                 
training/             


5. Future Improvements
• Increase dataset size for pose training
• Improve annotation consistency


6. Summary
This project demonstrates:
• End-to-end YOLOv8 workflow
• Transition from detection → classification → pose estimation
• Practical challenges of scaling model complexity with limited data
