import os
import csv
import random
import math
from PIL import Image
# import cv2
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model

def load_data(data_dir):
    """
    load the data based on the csv file from the given data directory
    :param data_dir: data directory, where images and mesurnments are stored
    return: train_samples, validation_samples
    """

    csv_path = data_dir + 'driving_log.csv'
    samples = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    
    #train_samples = [
    #    ['IMG/center_2016_12_01_13_32_43_457.jpg', 'IMG/left_2016_12_01_13_32_43_457.jpg', 'IMG/right_2016_12_01_13_32_43_457.jpg', 0.0617599, 0.9855326, 0, 2.124567], 
    #    ['IMG/center_2016_12_01_13_32_43_256.jpg', 'IMG/left_2016_12_01_13_32_43_256.jpg', 'IMG/right_2016_12_01_13_32_43_256.jpg', 0, 0, 0, 0.5024896],
    #    ['IMG/center_2016_12_01_13_32_46_185.jpg', 'IMG/left_2016_12_01_13_32_46_185.jpg', 'IMG/right_2016_12_01_13_32_46_185.jpg', -0.0787459, 0.9855326, 0, 29.91009]]
    #validation_samples = [
    #    ['IMG/center_2016_12_01_13_32_47_897.jpg', 'IMG/left_2016_12_01_13_32_47_897.jpg', 'IMG/right_2016_12_01_13_32_47_897.jpg', 0, 0.9855326, 0, 30.1871],
    #    ['IMG/center_2016_12_01_13_32_47_495.jpg', 'IMG/left_2016_12_01_13_32_47_495.jpg', 'IMG/right_2016_12_01_13_32_47_495.jpg', 0.05219137, 0.9855326, 0, 30.20486],
    #    ['IMG/center_2016_12_01_13_32_49_109.jpg', 'IMG/left_2016_12_01_13_32_49_109.jpg', 'IMG/right_2016_12_01_13_32_49_109.jpg', -0.05975719, 0.9855326, 0, 30.18686]]
    
    return train_samples, validation_samples


def generator(data_dir, samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            
            correction = 0.2 # this is a parameter to tune
            
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                center_name = data_dir + 'IMG/' + batch_sample[0].split('/')[-1]
                left_name = data_dir + 'IMG/' + batch_sample[1].split('/')[-1]
                right_name = data_dir + 'IMG/' + batch_sample[2].split('/')[-1]
                img_center = np.asarray(Image.open(center_name))
                img_left = np.asarray(Image.open(left_name))
                img_right = np.asarray(Image.open(right_name))

                # add images and angles (with flipped) to data set
                images.append(img_center)
                images.append(np.fliplr(img_center))
                images.append(img_left)
                images.append(np.fliplr(img_left))
                images.append(img_right)
                images.append(np.fliplr(img_right))

                angles.append(steering_center)
                angles.append(-steering_center)
                angles.append(steering_left)
                angles.append(-steering_left)
                angles.append(steering_right)
                angles.append(-steering_right)
                
                # name = data_dir + 'IMG/' + batch_sample[0].split('/')[-1]
                # center_image = mpimg.imread(name)
                # center_angle = float(batch_sample[3])
                # images.append(center_image)
                # angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def create_model():
    """
    create the model for training    
    """

    # Trimmed image format
    ch, row, col = 3, 160, 320

    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(row, col, ch)))
    # model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(row, col, ch)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    # 5x5 conv layer 1
    model.add(Conv2D(8, (5, 5), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))
    # 5x5 conv layer 2
    model.add(Conv2D(8, (5, 5), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))
    # fully connected layer 1
    model.add(Flatten())
    model.add(Dense(120))
    # fully connected layer 2
    model.add(Dense(84))
    # fully connected layer 3
    model.add(Dense(1))
    return model


def create_nvidia_model():
    """
    create the nvidia network from
    End to End Learning for Self-Driving Cars
    """
    # Trimmed image format
    ch, row, col = 3, 160, 320

    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(row, col, ch)))
    # model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(row, col, ch)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    # 5x5 conv layer 1
    model.add(Conv2D(24, (5, 5), activation='relu'))
    model.add(MaxPooling2D(padding='same'))
    # 5x5 conv layer 2
    model.add(Conv2D(36, (5, 5), activation='relu'))
    model.add(MaxPooling2D(padding='same'))
    # 5x5 conv layer 3
    model.add(Conv2D(48, (5, 5), activation='relu'))
    model.add(MaxPooling2D(padding='same'))
    # 3x3 conv layer 4
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # 3x3 conv layer 5
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    # fully connected layer 1
    model.add(Dense(100))
    model.add(Dropout(0.5))
    # fully connected layer 2
    model.add(Dense(50))
    model.add(Dropout(0.5))
    # fully connected layer 3
    model.add(Dense(10))
    model.add(Dropout(0.5))
    # fully connected layer 4
    model.add(Dense(1))
    return model

def train(train_samples, validation_samples, data_dir):
    # set the parameters
    batch_size = 32

    checkpoints = 'model.h5'

    # compile and train the model using the generator function
    train_generator = generator(data_dir, train_samples, batch_size=batch_size)
    validation_generator = generator(data_dir, validation_samples, batch_size=batch_size)
    print('length of training: ', len(train_samples))
    print('length of valid: ', len(validation_samples))
    
    # create the model for training
    model = create_nvidia_model()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # If trained model exist already then load first for further training
    if tf.gfile.Exists(checkpoints):
        model.load_weights(checkpoints)

    model.compile(loss='mse', optimizer='adam')
    

    early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(checkpoints, monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)

    model.fit_generator(train_generator,
                        steps_per_epoch=math.ceil(len(train_samples)/batch_size),
                        validation_data=validation_generator,
                        validation_steps=math.ceil(len(validation_samples)/batch_size),
                        epochs=5, 
                        verbose=1,
                        callbacks=[early_stop, checkpoint])

    # model.save(checkpoints)


if __name__ == '__main__':
    
    data_dir = '/opt/carnd_p3/data/'
    
    # Load data
    X_train, y_train = load_data(data_dir)
    # Train model
    train(X_train, y_train, data_dir)