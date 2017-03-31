# Self-Driving car-Behaviour Cloning
## Introduction
This project is a part of Udacitys self driving car nano degree.
The goal of the project is to train a deep learning network to learn and mimic the driving patterns of a human driver. Data is collected from a simulated driving on an animated track. Data is collected form of images as seen by an on borad camera system consiting of a center, left and right camera. The steering angle along with other parameters is recorded. The images forms the feature set for the CNN and the sterring angle is reponse variable. During the test phase, the car must be able to drive autonomously, under the direction of the CNN, i.e. the CNN must provide accurate steering angles in response to the images seen by camera during test.

## Network Architure
1. The network architecture [here] (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) was replicated
|Layer | Description|
| Input layer| 3x24|


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution (Conv1)      	|5x5 field, 1x1 stride, No padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution (Conv2)5x5	    | 5x5 field,1x1 stride, No padding, outputs 10x10x64      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Fully connected		(Fc0)| Input: 1600, output = 120        									|
| Fully connected		(Fc1)| Input: 120, output = 84        									|
| Fully connected		(Fc2)| Input: 84, output = 43        									|
| Softmax				|         									|

2. A dropout layer was added in front of every layer, with the dropout factor being a hyper parameter

## General discussion
1. There was no hard metric to optimze to.
  * The requirement was for the car to succesfully navigate. 
  * The target variable was the steering angle. How the MSE of the estimated steering angle would translate to a well behaved autonomous drive was not apparent.
2. Absent a concrete metric to optimize for, no particular effort was expended towards selection of a network, nor for fine tuning the parameters. 
  * Would a particular network hasve perfromed better than another? As with all things deep learning, this was not clear. 
  * In hingsight, this problem would have benifitted from a Recursive Neural network due to time component of the images. More discussion later.
3. This is seemingly a simple problem as the training and the test set is practically the same! So as long as the validation error fell towards zero, the car should have been self-drivable. 
 * However, given the sampling rate of the camera system, a large portion of the data contained small steering angles corresponding to straing driving or navigating longer curves. [Add figures]
 * These small driving angles overwhelmed the network leading to sturck steering.
3. Overall, this was an excercise in selecting the right training data
4. Things that should have been done
* Overtrain + drop out

## Methodology:
1. The network was first trained using one forward loop.
* Testing in autonomous mode created the struck steering. See [image] on the distribution.
* The near-zero steering, data were downsampled by a parametric ratio. A simple prametric threshold  was used to decide the range of steering that were removed
* The same thresold was used for both positive and negative steering. 
  * An argument  in favour of non symetric threshold can be made. However, the data was generated by driving the car in both directions, negating the use of non-symettric threshold
* Downsampling the data resulted in the sterring angle responding to road curvature. However it could not navigate the first turn.
2. Additional data was added to train the car to navigate the turns.  
* Data was generated by driving the car multiple times in eigther directions the region where it could not navigate
* The network was retrained after each additional data. See [figure for succesive addition of training data]. The distribution of the data 'looks'  identical, itrespective of which set of data is used. These CDFs were generated by throwing away 70% of data with steering angle less than 0.1. Changing the threshold 0.1 revals minor changes to CDF. This seemingly minor changes have a profound effect on the way the car drives.
* It can be concluded that the network depends on the few impactful training samples. This corroborates the view that adding more drives does not dramatically change the test behaviour.
* To test the thesis, the model was trained on two identical data, but each downsampled by a diffrent threhsold of 0.1 and 0.2. The later data set caused the car to over react. Clearly, it did not learn to drive straigh. 
* No correlation was found between the MSE and ability to drive. This again can be explained by the fact that plenty of straing driving data will lead to a small MSE but not lead toa successful drive.
3. The biggest impact was tighter cropping of the image. That along with adding dropouts and fine tuning the steering angle offset [explain] etc, reulted in  succesful mode.
#Augmentations: 
1. An approch based on generating a significant amount  data via augmentation failed. Augmentations included random brightness, shift along horizontal axis and...
* This is mostly expliable by the fact, that aumentation of useless data does not improve the quality of data. Since determing the correct ratio of good data
* I did not use any augmentation

####1. An appropriate model architecture has been employed
An Nvidia model as described in *link* was used 
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

The only parameter used was dropouts

#Train test split:
A 70-30 train test split was used, primarily to monitor the MSE


Closing Thoughts:
1.Treating the data as a time series sounds promising and will naturally address the small sterring angle problem.
2. A suggested, sub-optimal  architecure
* is to train a feed forward CNN 
* tap output of an intermidiate layer as a encoded feature vector.
* use the encoded features vectors as time training sequence of a Recucrsive NN.
3. A optimal solution is to train a feedforward network with RNN as one of the layers.
* The shallow feed forward layers will learn the discriminating features using that are relevant taking into account the time dependency.

Files in the project
model.py containing the script to create and train the model
drive.py for driving the car in autonomous mode
model.h5 containing a trained convolution neural network
writeup_report.md or writeup_report.pdf summarizing the results
####2. Submission includes functional code Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

python drive.py model.h5
####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy


####4. Appropriate training data
Track contains four turns.
Turn 1: before bridge, gentle
Turn 2: Immidiately after bridge. Sharp.
Turn 3: Immidiately past turn 2. Really sharp
Turn 4: Closer to the end of the lap.
Note that the MSE should be greater than 0.05- what does it mean?
Tried fine tuning steering angle  and downsampling factor
Decide to crop tigher. This was the key. Within a couple of attempts of fine tuning others, the car performed a complete lap.
Added additional turns. Did not work.

Augmented data:
Did not augment the original image.
Tried to have more robust solution-independent of the turns by augmentation
Added following aumentation, see code
Struck steering again. Could not resolve, gave up
Unfortuntely, the solution that worked was by trail and error.


Quatifying turn.
Overall, data for 2 compelete laps and multiple short turns was collecteed. 
For details about how I created the training data, see the next section.
Perform simulation
Join data framess
###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


