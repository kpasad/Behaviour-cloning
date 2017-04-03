import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import *
import cv2
from sklearn.utils import shuffle


import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda
from keras.layers import Cropping2D


def img_shift(img,x_pixels,y_pixels):    
   M = np.float32([[1,0,x_pixels],[0,1,y_pixels]])
   return(cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))) 

def img_bright_adjust(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv[:,:,2]=hsv[:,:,2]*np.random.uniform(0.3,1)
    return(cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR))
   
def aug_images(img,steering,params):
    x_shift = np.random.randint(params['x_shift_min'],params['x_shift_max'])
    img_x = img_shift(img,x_shift,0)
    steer_x = x_shift*params['steer_adjust_factor']
    img_b = img_bright_adjust(img_x)
    return(img_b,steer_x)

def generator(samples, batch_size,aug_params):
    num_samples = len(samples)
    img_idx=0    
    right_steering_offset=-0.2
    left_steering_offset=0.2
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        images=[]
        angles=[]
        batch_img_cnt=0
        while batch_img_cnt < batch_size:
            #print("Processing image {}".format(img_idx))
            center_name = samples['center'][img_idx].strip()
            right_name  = samples['right'][img_idx].strip()
            left_name   = samples['left'][img_idx].strip()

            if not os.path.isfile(center_name.strip()):
                print(center_name)
            if not os.path.isfile(right_name.strip()):
                print(right_name)
            if not os.path.isfile(left_name.strip()):
                print(left_name)
            center_image = cv2.imread(center_name)
            center_angle = float(samples['steering'][img_idx])
            images.append(center_image)
            angles.append(center_angle)
                
            for replica_cnt in range(aug_params['aug_factor']-1):
                replica,steering=aug_images(center_image,center_angle,aug_params)
                images.append(replica)
                angles.append(steering)
                batch_img_cnt+=1
                #print("Adding {} image to the batch".format(batch_img_cnt))
            img_idx=np.mod(img_idx+1,num_samples)   
            
        X_train = np.array(images)
        y_train = np.array(angles)
        yield shuffle(X_train, y_train)

def nvidia_model():
    nb_filt_conv1 = 24
    nb_filt_conv2 = 36
    nb_filt_conv3 = 48
    nb_filt_conv4 = 64
    nb_filt_conv5 = 64
    
    fc1_op = 1164
    fc2_op = 100
    fc3_op = 50
    fc4_op = 10
    fc5_op = 1
    kernel_1 = 5
    kernel_2 = 3
    
    stride_1=2
    actv='relu'
    #wt_init = 'glorot_uniform'
    wt_init = 'he_normal'
    
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))


    model.add(Convolution2D(nb_filt_conv1, kernel_1, kernel_1, subsample=(stride_1,stride_1),init=wt_init,border_mode='valid',
    	                        input_shape=(60,280,3)))
    model.add(Activation(actv))
    
    
    model.add(Convolution2D(nb_filt_conv2, kernel_1, kernel_1, subsample=(stride_1,stride_1),init=wt_init,border_mode='valid'))
    model.add(Activation(actv))
    
    model.add(Convolution2D(nb_filt_conv3, kernel_2, kernel_2, subsample=(stride_1,stride_1),init=wt_init,border_mode='valid'))
    model.add(Activation(actv))
    
    model.add(Convolution2D(nb_filt_conv4, kernel_2, kernel_2, subsample=(1,1),init=wt_init,border_mode='valid'))
    model.add(Activation(actv))
    
    model.add(Convolution2D(nb_filt_conv5, kernel_2, kernel_2, subsample=(1,1),init=wt_init,border_mode='valid'))
    model.add(Activation(actv))
    
    model.add(Flatten())
    
    model.add(Dense(fc1_op,init=wt_init))
    model.add(Activation(actv))
    
    model.add(Dense(fc2_op,init=wt_init))
    model.add(Activation(actv))
    
    model.add(Dense(fc3_op,init=wt_init))
    model.add(Activation(actv))
    
    model.add(Dense(fc4_op,init=wt_init))
    model.add(Activation(actv))
    
    model.add(Dense(fc5_op,init=wt_init))
    return model

def test_train_split_df(full_feats,cv_ratio):
    n_total = len(full_feats)
    n_train = round(n_total*cv_ratio)
    train_feats = full_feats[0:n_train]
    test_feats   = full_feats[n_train+1:n_total]
    return(train_feats,test_feats.reset_index())

def balance_df(df,downsample_factor) :
    sample_idx = shuffle(np.where(df['steering']==0)[0])    
    drop_idx = sample_idx[0:round(len(sample_idx)*downsample_factor)]
    balanced_data = df.drop(drop_idx)
    return balanced_data.reset_index()
    


basepath = r'C:\Users\kpasad\mydata\sdc\p3_data'
basepath= basepath.replace('\\','/')
feature_file = basepath+'/multi_track_win.csv'

#basepath='/media/kalpendu/BACKUP/ml/data/sdc/p3_data'
#feature_file = basepath+'/multi_track_lnx.csv'



full_feats = read_csv(feature_file)

downsample_factor=0.9
balanced_df=balance_df(full_feats,downsample_factor)

aug_params ={}
aug_params['aug_factor'] =5
aug_params['x_shift_min'] =-30
aug_params['x_shift_max'] = 30
aug_params['steer_adjust_factor'] = 0.004

cv_ratio=0.9
batch_size=32
train_feats,test_feats = test_train_split_df(balanced_df,cv_ratio)
train_feats=train_feats[0:int(np.floor(len(train_feats)/batch_size)*batch_size)]
train_generator = generator(train_feats,batch_size,aug_params)


model=nvidia_model()# Preprocess incoming data, centered around zero with small standard deviation 
model.compile(loss='mse', optimizer='adam')
history_object=model.fit_generator(train_generator, samples_per_epoch=len(train_feats)*3, nb_epoch=1,verbose=1)

with open('model_lnx.json', 'w') as fd:
   json.dump(model.to_json(), fd)
model.save('model_win.h5')
test_feats=test_feats[0:int(np.floor(len(test_feats)/batch_size)*batch_size)]

test_generator = generator(test_feats,32,aug_params)

pred_steer = model.predict(test_generator,batch_size=32)

score = model.evaluate_generator(test_generator,val_samples=len(test_feats))
print('Test score:',score)
