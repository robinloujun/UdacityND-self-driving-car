# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

- First, I converted the images to grayscale with `cv2.cvtColor()` 
- Then I applied Gaussian blur to the grayscale the Canny detection is uesed to generate all edges.
- To figure out which edges are more likely to be the lane lines, a region of interest is masked.
- The long lane lines are generated with help of Hough transformation with parameter tuning.
- Finally the original image and the extracted lane lines are merged into one image and outputed.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by computing the slope of extracted line segments. In order to get rid of the shadow on the road, only the line segments with slope between 30 and 60 degrees are regarded as lane line candidates. And the line segmemnts with positive slope are set to left lane line and negative set to right. Then the two lane lines are interpolated with the segments.

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the lane is not so clear, which happens in the challenge.mp4. The lane seem to depature from the ground-truth.

Another shortcoming could be the vibration of the deteced lane lines because the lines deteced in each frame are only based on the information from current frame.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use the consecutive frames to detect the lane lines, like using a median filter for the lines detected with former frames.
