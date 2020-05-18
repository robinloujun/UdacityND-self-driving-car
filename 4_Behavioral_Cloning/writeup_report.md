# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./imgs/nvidia.png "Reference CNN architecture"
[image1]: ./imgs/model_plot.png "Model Visualization"
[image2]: ./imgs/center.jpg "Center Lane Driving"
[image3]: ./imgs/recovery_1.jpg "Recovery Image"
[image4]: ./imgs/recovery_2.jpg "Recovery Image"
[image5]: ./imgs/recovery_3.jpg "Recovery Image"
[image6]: ./imgs/center.jpg "Normal Image"
[image7]: ./imgs/center_flip.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 recording one and a half laps of the autonomous driving mode

#### 2. Submission includes functional code
Using the model.py file, the dataset can be read and trained with the CNN and the checkpoint can be saved as model.h5 by executing
```sh
python model.py
```

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network which is constructed from the [reference](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), it is work at Nvidia Corporation. The original architecture is 

![alt text][image0]

The model includes RELU layers to introduce nonlinearity (code line 146), and the data is normalized in the model using a Keras lambda layer (code line 144). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 161, 164 & 167). 

Besides I also applied data augumentation by adding the cropped frames into training dataset to reduce overfitting (model.py lines 75-87).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 33). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 191). The batch size is set at 32 and the epoch number is set at 5. The final loss is around 0.029. After that there is no improvement by adding the epochs so 5 is a proper parameter.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet, I thought this model might be appropriate because it works really well in the traffic sign classification project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, typically at the curve with dirt boundary after the bridge. To improve the driving behavior in these cases, I added several recovery sequences in the specific segments and throw the data into training. 

To combat the overfitting and also the failure at the specific position, I turned to the nvidia network, thanks to the [advice of Paul](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Behavioral+Cloning+Cheatsheet+-+CarND.pdf). The training and tuning process is exactly the same as before and at the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is defined in model.py lines 132-170.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

The overview of all data:

- two laps of center lane driving at the rate of 10 mph, 20 mph and 25 mph in initial direction.
- two laps of center lane driving at the rate of 10 mph, 20 mph and 25 mph in opposite direction.
- multiple sequences of recovery driving from the sides
- multiple sequences focusing on driving smoothly around curves
- multiple sequences focusing on driving around the dirt boundary after the bridge

To capture good driving behavior, I first recorded two laps on track one using center lane driving at the rate of 10 mph, 20 mph and 25 mph, then recored the same tracks in the opposite direction. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the right side and right sides of the road back to center so that the vehicle would learn to correct the maneuvers. These images show what a recovery looks like starting from the right side:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would reduce the overfitting. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 28379 number of data points. By flipping and append the left and right cameras the number of samples was 170274.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced that the loss does not decrease that much. Besides I applied the callback by saving the best model only. I used an adam optimizer so that manually training the learning rate wasn't necessary.
