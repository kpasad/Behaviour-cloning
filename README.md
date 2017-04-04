# Self-Driving car-Behaviour Cloning
## Introduction
This project is a part of Udacitys self driving car nano degree.
The goal of the project is to train a deep learning network to learn and mimic the driving patterns of a human driver. Data is collected from a simulated driving on an animated track. Data is collected in form of images as seen by an on borad camera system consiting of a center, left and right camera. The steering angle along with other parameters are recorded. The images form the feature set for the a deep learning convolutional neural network and the sterring angle is reponse variable. During the test phase, the car must be able to drive autonomously, under the direction of the CNN, i.e. the CNN must provide accurate steering angles in response to the images seen by camera during test.

## Network Architure
1. The network architecture [here] (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) was replicated



| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 90,320,3 RGB color image   							| 
| Convolution (Conv1)      	|5x5 field, 2x2 stride, No padding, outputs 43x158x24 	|
|Dropout | 0.8|
| RELU					|												|
| Convolution (Conv2)	    | 5x5 field,2x2 stride, No padding, outputs 20x77x36      									|
|Dropout | 0.8|
| RELU					|												|
| Convolution (Conv3)	    | 3x3 field,2x2 stride, No padding, outputs 9x38x48      									|
|Dropout | 0.8|
| RELU					|												|
| Convolution (Conv4)	    | 3x3 field,2x2 stride, No padding, outputs 7x36x64      									|
|Dropout | 0.8|
| RELU					|												|
| Convolution (Conv4)	    | 3x3 field,2x2 stride, No padding, outputs 5x34x64      									|
|Dropout | 0.8|
| RELU					|												|
| Fully connected		(Fc1)| Input: 10880, output = 1164        									|
| Fully connected		(Fc2)| Input: 1164, output = 100        									|
| Fully connected		(Fc3)| Input: 100, output = 50        									|
| Fully connected		(Fc4)| Input: 50, output = 10        									|
| Fully connected		(Fc5)| Input: 10, output = 1        									|


## Preprocessing
1. Images were centered (zero mean)
2. Original images sized 160x320 were cropped to 90x320
Both preprocessing steps were implemented in keras using Lamda layer followed by cropping layer

## Data generation
1. The images from simulator are stored on the hard drive. Loading all of them is not possible due to memory constraints. So a genertor function is utilised. See code for more details
2. Images from all the three cameras, center, left and right are used
3. The images from left and right camera are angularly skewed. Any estimation of steering angle based on the side cameras should be offset. The offset varies according to road curvature in deterministic manner. While a trignometric relationship could be derived, a fixed offset was used in this effort.
4. A 80-20 train test split was used, primarily to monitor the MSE


## Methodology and observations:
1. The network was first trained using images from one forward loop along the simulated track.
* Testing the network in autonomous mode created the 'struck-steering', a condition where the CNN does not generate any steering correction  in reponse to a curvature in the road. 
* This happens because the training data is imbalanced with much larger proportion of images correponding to a sterring angle of close to zero. The dominance of small steering angle is apparent in the PDF of steering angles for one lap around the track ![PDF of imbalanced data set](/images/pdf.png) 
* To create a balanced data set, the near-zero steering data were downsampled. A simple prametric threshold  was used to decide the range of steering that were removed. The PDF for of same data set as above, but with 30% of steering angles less than absolute 0.1 thrown away is shown below ![PDF of balanced data set](/images/pdf_70perc_point1.png). The same thresold was used for both positive and negative steering. An argument  in favour of non symetric threshold can be made. However, eventually, the data was generated by driving the car in both directions, negating the use of non-symettric threshold. 
* Note that suppressing smaller steering angles skews the means but retains the variance. e.g the following table is for two complete loops around the track; one if forward direction and one in reverse direction. The balanced data is created by throwing away 30 percent of steering angles less than 0.1. There is a minor impact on variance.  The change in mean value is minor in forwar direction. Note that the mean steering values in reverse direction are not the mirror value of forward drive. The mean value is infact biased, perhaps a reflection of driving habit. Downsampling thus skews the reverse drive more than forward

|Drive direction| imbalance mean | balanced mean | imbalance variance| balance variance|
|:--:|:--:|:--:|:--:|:--:|
|Fwd|-0.035|-0.048|0.01489|0.02628|
|Rev|-0.2791|0.05390|0.0118|0.0229|

2. Downsampling the data resulted in the steering angle responding to road curvature. However it could not navigate the turns.
3. To correct for it, additional sata was generated by driving the car multiple times in eigther directions the region where it could not navigate. The network was retrained after each additional data. Figure below shows the CDF of each component drive ![CDF of steering angles for a variety of drives](/images/cdf_drives.png).
* It also intresting to see how the CDF of various drives changes as the downsampling factor is changed.![CDF of individual drives with diffrent downsampling ](/images/cdf_drives_ds.png)

* Figure below shows the CDF of combined drive ![CDF of combined drives data ](/images/cfd_successive_additions.png) .
* While the distribution of each drive looks diffrent, when the data is combined into a single drive, each data set seems indistinguishable.
4. Some notes 
 * These CDFs were generated by throwing away 70% of data with steering angle less than 0.1. Downsampling creates only a minor changes to CDF. This seemingly minor changes, however have a profound effect on the way the car drives.
 * Since the combined data is shuffled, a question arises: will the network benifit from training individually on each data set sequntially using tranfer learning? 
 * It can be concluded that the network depends on the few impactful training samples. This corroborates the observation that adding more drives does not dramatically change the test behaviour, as most data corresponds to a small streeting angle. To test the thesis, the model was trained on two identical data, but each downsampled by a diffrent threhsold of 0.1 and 0.2. The later data set caused the car to over react to minor turns. It had difficulty driving straight. A sweet spot of data distribution exisits, between threshold of 0.1 and 0.2
 * No correlation was found between the MSE and ability to drive except for a approximate minimum MSE. This again can be explained by the fact that plenty of straing driving data will lead to a small MSE but not lead toa successful drive.
5. Overall, this was an excercise in selecting the right training data
6. The biggest impact was tighter cropping of the image. After some experimentation with diffrent data, a set of parameters waas obtained that resulted in the car driving itself succesfully.

## Network training
1. No effort was expended in trying out diffrent architecures. During the initial phases, there was no a apparent relation between MSE of the estimates and the vehicles ability to drive autonomously.
2. It was clear though, that the test MSE needed to be less than 0.03 for the car to navigate correctly. The variance of the steering angle is ~0.025. A mean square greater that 0.025 would mean that predictions have more randomness then the data. However, this aument assumes that the data is unimodal. The sterring angle should be viewed as a multimodal distribution. E.g. A curvature of certain radii, corresponds to a mean sterring angle and a varaince around it. The mean shifts with the radii, creating a multi-modal continous distribution.
3. Training was limited to ~4 epochs. Within this duration validation error reached a steady state.
4. The training data that worked consisted of one forward, one reverse, and one multiple passes of the shrpest curves
5. Training data that failed were when data for one specific curve occured in a significant number.

## Augmentations: 
1.Augmentations included random brightness, shift along horizontal axis. 
An approch based on generating a significant amount  data via augmentation failed.
* This is mostly expliable by the fact, that aumentation of useless data does not improve the quality of data. Since determing the correct ratio of good data
* I did not use any augmentation
* Augmentation allows a tighter control of statistics of training data.With proper augmentation, a much smaller data set could be used.

## Closing Thoughts:
1.Treating the data as a time series sounds promising and will naturally address the small sterring angle problem.
2. A suggested, sub-optimal  architecure
* is to train a feed forward CNN 
* tap output of an intermidiate layer as a encoded feature vector.
* use the encoded features vectors as time training sequence of a Recucrsive NN.
3. A optimal solution is to train a feedforward network with RNN as one of the layers.
* The shallow feed forward layers will learn the discriminating features using  taking into account the time dependency.

## Files in the project:
* train_drive_data.py:script to create and train the model based on data collected from simulated driving only.
* train_aug_data.py  :script to create and train the model based on data collected from simulated driving as well as augmented data.
* join_tracks.py : Script to selective combine data from diffrent drives.
* drive.py : Created and provided by udacity for driving the car in autonomous mode
* model.h5:  containing a trained convolution neural network. It is downloadable from:

[Trained Keras Model](https://www.dropbox.com/s/kekjrl564kf08x9/model.h5?dl=0)
* run1.mp4: Video of car driving autonomously, It is downloadable from:

[Video of autonomous driving](https://www.dropbox.com/s/21zhgxe7l2xtgv4/run1.mp4?dl=0)
