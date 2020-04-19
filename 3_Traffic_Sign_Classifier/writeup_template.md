# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_imgs/sample_visualization.png "Random Data Visualization"
[image1_1]: ./output_imgs/label_hist.png "Label Histogram Visualization"
[image2]: ./output_imgs/gray_norm.png "Grayscaling and Normalization (Training)"
[image2_1]: ./output_imgs/gray_norm_test.png "Grayscaling and Normalization (Testing)"
[image3]: ./output_imgs/ground_truth.jpg "Ground Truth"
[image4]: ./output_imgs/prediction.png "Prediction"
[image5]: ./output_imgs/softmax_prob.png "Softmax Probabilities"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the function `len()` calculate summary statistics of the traffic signs data set. When counting the number of classes, I used `np.unique()` from Numpy library:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set, which visualize 30 images with the class index randomly. After this I produced three bar charts showing how the data contribute in training, validation and testing dataset with `np.histgram()` from Numpy library:

![alt text][image1]
![alt text][image1_1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color information of traffice signs is not necessary for classification. Besides, I normalized the image data to a range of /[-1.0, 1.0/] because this can get rid of the influence of grayscale distribution.

Here is examples of five traffic sign images before and after grayscaling & normalization from training and testing dataset.

![alt text][image2]
![alt text][image2_1]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image						| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x128	 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x256 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x256 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4X4X256	 				|
| Flatten				| outputs 4096									|
| Fully connected		| outputs 400									|
| Fully connected		| outputs 200									|
| Fully connected		| outputs 43									|
| Softmax				| etc.        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an customized CNN structure according to VGG-16, which is defined in cell 12. The type of optimizer is AdamOptimizer, the batch is 128, the number of epochs is 20, the dropout ratio is 0.5 and the learning rate is 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.974 
* test set accuracy of 0.959

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I first tried with the LeNet, because it performs very well for the MNIST dataset. And we do not need to change much to the existing structure, the only change to LeNet-Lab is the number of output neurons from 10 to 43 to fit the application. 

* What were some problems with the initial architecture?

The neural network might not be deep enough so the validation and test accuracy is a bit lower. When applying high epoch number, the validation accuracy can be near 0.93.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

First I added dropout to the LeNet, which is defined as the function `LeNet_modified` in the cell. The dropout ratio is 0.5. When applying high epoch number, the validation accuracy could reach 0.95, but test accuracy about 0.925.

So I chose the VGG-16 as the final solution, this CNN structure is also very famous, and the designed structure is listed in point 2. With this structure, we can get higher accuracy with less epochs.

* Which parameters were tuned? How were they adjusted and why?

I tuned the epoch number, batch size and learning rate. For learning rate I did not find much improvement in accuracy so I fixed it at 0.001. The batch size has a large influence to the training rate and also the accuracy and 128 is a usual choise and it is also proper for this sturcture. For the required accuracy 0.93, the VGG network can reach this requirment at about 10 epochs but it should be chosen according to the highest point before overfitting.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?

As I mentioned in the last question, I chose the VGG-16 as the final solution.

* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
From the final result we can see that the VGG works very well in this application. I guess that VGG works even better for more complex scenarios but for the cases in traffice signs it is a good tool.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I chose from the testing dataset:

![alt text][image3] 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image								|     Prediction	        					| 
|:---------------------------------:|:---------------------------------------------:| 
| Vehicle over 3.5 meter			| Vehicle over 3.5 meter						| 
| Speed limit (30 km/h)				| Speed limit (30 km/h)							|
| Keep right						| Keep right									|
| Turn right ahead 					| Turn right ahead				 				|
| Right-of-way at the next junction	| Right-of-way at the next junction				|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.9%.

![alt text][image4]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 100%), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.0%       			| Vehicle over 3.52 meter						| 
| 0.0%     				| Speed limit (20 km/h) 						|
| 0.0%					| Speed limit (80 km/h)							|
| 0.0%	      			| No passing					 				|
| 0.0%				    | Roundabout mandatory 							|

A summary of outputs is like following:
![alt text][image5]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


