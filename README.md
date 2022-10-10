# Using Deep Learning in Ultrasound Imaging of Bicipital Peritendinous Effusion to Grade Inflammation Severity

##  • Architecture of the proposed system
![imag e](https://user-images.githubusercontent.com/39873770/194925141-168415ad-84af-4b5a-9029-76d0fe60b527.png)

##  • Constructed a faster R-CNN model 
To automatically determine the region of interest in ultrasound images to reduce 99% manual identification
![image](https://user-images.githubusercontent.com/39873770/194925248-6fc25775-debc-4879-b960-942ace61ca44.png)

Original ultrasound images contain redundant information around the biceps area, which reduces classification accuracy. To eliminate redundant information and retain biceps containing regions, the faster R-CNN method is used to automatically determine the ROI.

##  • Data Preprocessing
![image](https://user-images.githubusercontent.com/39873770/194943175-fd63b3d4-940b-40b0-872b-20df88033543.png)

(a) Target image. (b) Original image. (c) Image obtained after normalization

Inconsistencies such as different deflection angles, depths, and brightness may be present in ultrasound images. These inconsistencies can influence classification performance. Therefore, image normalization should be conducted to address such inconsistencies. The normalized image is shown to be clearer than the original image.
 
