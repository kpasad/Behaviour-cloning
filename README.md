# Self-Driving car-Behaviour Cloning
## Introduction
This project is a part of Udacitys self driving car nano degree.
The goal of the project is to train a deep learning network to learn and mimic the driving patterns of a human driver. Data is collected from a simulated driving on an animated track. Data is collected form of images as seen by an on borad camera system consiting of a center, left and right camera. The steering angle along with other parameters is recorded. The images forms the feature set for the CNN and the sterring angle is reponse variable. During the test phase, the car must be able to drive autonomously, under the direction of the CNN, i.e. the CNN must provide accurate steering angles in response to the images seen by camera during test.

## Network Architure
1. The network architecture [here] (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) was replicated



| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 60,280,3 RGB color image   							| 
| Convolution (Conv1)      	|5x5 field, 2x2 stride, No padding, outputs ?x?x24 	|
| Maxpool| 2x2 Stride |
|Dropout | 0.8|
| RELU					|												|
| Convolution (Conv2)	    | 5x5 field,2x2 stride, No padding, outputs ?x?x36      									|
| Maxpool| 2x2 Stride |
|Dropout | 0.8|
| RELU					|												|
| Convolution (Conv3)	    | 3x3 field,2x2 stride, No padding, outputs ?x?x48      									|
| Maxpool| 2x2 Stride |
|Dropout | 0.8|
| RELU					|												|
| Convolution (Conv4)	    | 3x3 field,2x2 stride, No padding, outputs ?x?x64      									|
|Dropout | 0.8|
| RELU					|												|
| Convolution (Conv4)	    | 3x3 field,2x2 stride, No padding, outputs ?x?x64      									|
|Dropout | 0.8|
| RELU					|												|
| Fully connected		(Fc1)| Input: ?, output = 1164        									|
| Fully connected		(Fc2)| Input: 1164, output = 100        									|
| Fully connected		(Fc3)| Input: 100, output = 50        									|
| Fully connected		(Fc4)| Input: 50, output = 10        									|
| Fully connected		(Fc5)| Input: 10, output = 1        									|


## Preprocessing
1. Images were centered (zero mean)
2. Original images sized 160x320 were cropped to 70x280
Both preprocessing steps were implemented in keras

## Data generation
1. The images from simulator are stored on the hard drive. Loading all of them is not possible due to memory constraints. So a genertor function is utilised. See code for more details
2. The images from left and right camera are angularly skewed. Any estimation of steering angle based on the side cameras should be offset. The offset varies according to road curvature in deterministic manner. While a trignometric relation ship could be derived, a fixed offset was used.
3. A 80-20 train test split was used, primarily to monitor the MSE


## Methodology:
1. The network was first trained using one forward loop along the simulated track.
* Testing in autonomous mode created the 'struck-steering', a condition where the CNN does not generate any steering correction  in reponse to a curvature in the road. 
* This happens because the training data is imbalanced with much larger proportion of images correponding to a sterring angle of zero. See the following PDF and CDF of steering angle for one lap around the track ![PDF of imbalanced data set](/images/pdf.png) 
* To create a balanced data set, the near-zero steering data were downsampled. A simple prametric threshold  was used to decide the range of steering that were removed. The same thresold was used for both positive and negative steering. An argument  in favour of non symetric threshold can be made. However, the data was generated by driving the car in both directions, negating the use of non-symettric threshold. 
* Suppressing smaller steering angles skews the Means but retains the variance. e.g the following table is for two complete loops around the track; one if forward direction and one in reverse direction. The balanced data is created by throwing away 30 percent of steering angles less than 0.1. There is a minor impact on variance.  The chane in means is minor in fwd direction, but dramatic in reverse direction. This is not surprising since the sterring in reverse directions has biased in positive side.

|Drive direction| imbalance mean | balanced mean | imbalance variance| balance variance|
|:--:|:--:|:--:|:--:|:--:|
|Fwd|-0.035|-0.048|0.01489|0.02628|
|Rev|-0.-2791|0.05390|0.0118|0.0229|
* Downsampling the data resulted in the sterring angle responding to road curvature. However it could not navigate the first turn.
2. To correct for it, additional data was added to train the car to navigate the turns.  
See ![CDF of steering angles for a variety of drives](/images/cdf_drives.png).


* Data was generated by driving the car multiple times in eigther directions the region where it could not navigate.
* The network was retrained after each additional data. ![CDF of combined drives data ](/images/cfd_successive_additions.png)
* While the distribution of each drive looks diffrent, when the data is combined into a single drive, each data set seems indistinguishable.
* There are two observations
 * These CDFs were generated by throwing away 70% of data with steering angle less than 0.1. Changing the threshold 0.1 revals minor changes to CDF. This seemingly minor changes have a profound effect on the way the car drives.
 * will the network benifit from training individually on each data set sequntially using *weiht tranfer* 
 * It can be concluded that the network depends on the few impactful training samples. This corroborates the observation that adding more drives does not dramatically change the test behaviour, as most data corresponds to a small streeting angle
 * To test the thesis, the model was trained on two identical data, but each downsampled by a diffrent threhsold of 0.1 and 0.2. The later data set caused the car to over react. Clearly, it did not learn to drive straigh. 
 * No correlation was found between the MSE and ability to drive. This again can be explained by the fact that plenty of straing driving data will lead to a small MSE but not lead toa successful drive.
3. Overall, this was an excercise in selecting the right training data

4. The biggest impact was tighter cropping of the image.

#Augmentations: 
1.Augmentations included random brightness, shift along horizontal axis. 
An approch based on generating a significant amount  data via augmentation failed.
* This is mostly expliable by the fact, that aumentation of useless data does not improve the quality of data. Since determing the correct ratio of good data
* I did not use any augmentation





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


