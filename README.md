Using Deep Learning in Ultrasound Imaging of Bicipital Peritendinous Effusion to Grade Inflammation Severity

#### Architecture of the proposed system
![image](https://user-images.githubusercontent.com/39873770/194925141-168415ad-84af-4b5a-9029-76d0fe60b527.png)

#### Constructed a faster R-CNN model to automatically determine the region of interest in ultrasound images to reduce 99% manual identification
![image](https://user-images.githubusercontent.com/39873770/194925248-6fc25775-debc-4879-b960-942ace61ca44.png)

Original ultrasound images contain redundant information around the biceps area, which reduces classification accuracy. To eliminate redundant information and retain biceps containing regions, an ROI detection model can be used to select image regions that depict BPE
