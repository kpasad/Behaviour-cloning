# Self-Driving car-Behaviour Cloning
## Introduction
This project is a part of Udacitys self driving car nano degree.
The goal of the project is to train a deep learning network to learn and mimic the driving patterns of a human driver. Data is collected from a simulated driving on an animated track. Data is collected form of images as seen by an on borad camera system consiting of a center, left and right camera. The steering angle along with other parameters is recorded. The images forms the feature set for the CNN and the sterring angle is reponse variable. During the test phase, the car must be able to drive autonomously, under the direction of the CNN, i.e. the CNN must provide accurate steering angles in response to the images seen by camera during test.

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

###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

model.py containing the script to create and train the model
drive.py for driving the car in autonomous mode
model.h5 containing a trained convolution neural network
writeup_report.md or writeup_report.pdf summarizing the results
####2. Submission includes functional code Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

python drive.py model.h5
####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The overall approch of the :
1. There was hard metric to optimze to. the requirement was for the car to succesfully navigate
2. How the MSE would translate to good driving was not clear
3. Avoided fine tuning and model selection

Used the nvidia model with following parameters
The only parameter used was dropouts

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data
Track contains four turns.
Turn 1: before bridge, gentle
Turn 2: Immidiately after bridge. Sharp.
Turn 3: Immidiately past turn 2. Really sharp
Turn 4: Closer to the end of the lap.
1. Began with single front track. 
2. Struck steering
2a. Overfitting to the straing
2b. Experiment with removing straights.
2c. Can navigate past the first bend
2d. Add reverse path
2e. Add additional camera with 0.2
Cannot navigate Turn2.
Simulate turn2- repeated back and forths around turn 2
Turn 2 is successful but cannot navigate turn 3.
Add more training data around turn 3 but cannot navigate.
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

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

alt text

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

alt text

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

alt text alt text alt text

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

alt text alt text

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...

I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
Contact GitHub API Training Shop Blog About
© 2017 GitHub, Inc. Terms Privacy Security Status Help

