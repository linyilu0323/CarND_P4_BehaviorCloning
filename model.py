## import some necessary libraries
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.regularizers import l2
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import ceil
from os import getcwd
from os import path

## define some helper functions

def get_driving_log_entries(data_path):
    '''
    get image file paths and steering angles from driving log
    '''
    log_file_path = data_path + 'driving_log.csv'
    #img_file_path = data_path + '/IMG/'
    image_paths = []
    steering_angles = []
    with open(log_file_path, 'r') as f:
        drive_log_entries = csv.reader(f, skipinitialspace=True)
        next(drive_log_entries, None)
        for entry in drive_log_entries:
            image_fn = path.split(entry[0])[1]
            image_paths.append(data_path + '/IMG/' + image_fn)
            steering_angles.append(entry[3])
            ## adding both sides of camera, defining angle offset
            angle_cor = 0.25
            # left camera image
            image_fn = path.split(entry[1])[1]
            image_paths.append(data_path + '/IMG/' + image_fn)
            steering_angles.append(float(entry[3]) + angle_cor)
            # right camera image
            image_fn = path.split(entry[2])[1]
            image_paths.append(data_path + '/IMG/' + image_fn)
            steering_angles.append(float(entry[3]) - angle_cor)

    image_paths = np.array(image_paths)
    steering_angles = np.array(steering_angles).astype(float)

    return image_paths, steering_angles

def generate_training_data(paths, angles, batch_size=128):
    '''
    implementation for training data generator
    '''
    while True:
        paths, angles = shuffle(paths, angles)
        for offset in range(0, len(angles), batch_size):
            inputs = []
            outputs = []
            batch_paths = paths[offset:offset+batch_size]
            batch_angles = angles[offset:offset+batch_size]
            for (img_fn, angle) in zip(batch_paths, batch_angles):
                img = cv2.imread(img_fn)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                inputs.append(img)
                outputs.append(angle)
                # flip image horizontally and add them to training set
                inputs.append(cv2.flip(img,1))
                outputs.append(-1*angle)

            inputs = np.array(inputs)
            outputs = np.array(outputs)
            yield shuffle(inputs, outputs)

def preprocess_population(X, y, num_bins=21):
    hist, bins = np.histogram(y, num_bins)
    trim_threshold = len(y) / num_bins
    # trim any over-represented group to the trim_threshold (here: average population).
    X, y = shuffle(X, y)
    remove_inds = []
    for i in range(len(hist)):
        count = 0
        if hist[i] > trim_threshold:
            for j in range(len(y)):
                if (y[j] > bins[i]) & (y[j] < bins[i+1]):
                    count += 1
                    if count > trim_threshold:
                        remove_inds.append(j)
    X_out = np.delete(X, remove_inds, axis=0)
    y_out = np.delete(y, remove_inds)
    return X_out, y_out

def visualize_image(image, angle, angle_pred=None):
    '''
    Plot the image (input needs to be in RGB format)
    Annotate the image with the steering angle from training data,
    If exist, also annotate the image with the model predicted steering angle.
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.25
    fontColor = (255, 255, 255)
    fontColor_pred = (255, 0, 0)
    thickness = 1

    str_input = 'Angle = {:1.4f}'.format(angle)
    image = cv2.putText(image, str_input, (20, 20), font, fontScale, fontColor, thickness)

    if angle_pred is not None:
        str_pred = 'Angle = {:1.4f}'.format(angle_pred)
        image = cv2.putText(image, str_pred, (20, 50), font, fontScale, fontColor_pred, thickness)

    plt.imshow(image)
    plt.show()

###########################
### Main program
###########################

## load raw data
image_paths_raw, steering_angles_raw = get_driving_log_entries(getcwd()+'/data/')
print('Total Number of Images Imported: {}'.format(len(steering_angles_raw)))

## pre-process data to trim over-representing outputs
num_bins = 21
image_paths, steering_angles = preprocess_population(image_paths_raw, steering_angles_raw, num_bins=num_bins)

## visualize training data before and after preprocessing
# make a bar chart to show the distribution of steering angle vs. number of images
hist_raw, bins = np.histogram(steering_angles_raw, num_bins)
hist, bins = np.histogram(steering_angles, num_bins)
width = 0.35 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.figure(figsize=(8,3))
plt.grid(which='both', axis='x')
plt.bar(center-width*0.3, hist_raw, width=width, color='red')
plt.bar(center+width*0.3, hist, width=width, color='blue')
plt.ylabel('# of images')
plt.xlabel('Steering Angle')
plt.show()

print('Total Number of Images Now: {}'.format(len(steering_angles)))

## compile the final training data
# split into training and validation sets
img_fn_train, img_fn_valid, steering_angle_train, steering_angle_valid = \
    train_test_split(image_paths, steering_angles, test_size=0.2)

print('Total Number of Images in Training set: {}'.format(len(steering_angle_train)))
print('Total Number of Images in Validation set: {}'.format(len(steering_angle_valid)))

# initialize generators
bs = 128
train_generator = generate_training_data(img_fn_train, steering_angle_train, batch_size=bs)
valid_generator = generate_training_data(img_fn_valid, steering_angle_valid, batch_size=bs)

## model architecture: nVidia model, by Bojarski et al.
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Conv2D(24, kernel_size=(5,5), subsample=(2,2), activation='relu', kernel_regularizer=l2(0.001)))
model.add(Conv2D(36, kernel_size=(5,5), subsample=(2,2), activation='relu', kernel_regularizer=l2(0.001)))
model.add(Conv2D(48, kernel_size=(5,5), subsample=(2,2), activation='relu', kernel_regularizer=l2(0.001)))
model.add(Conv2D(64, kernel_size=(3,3), subsample=(2,2), activation='relu', kernel_regularizer=l2(0.001)))
model.add(Conv2D(64, kernel_size=(3,3), subsample=(2,2), activation='relu', kernel_regularizer=l2(0.001)))
model.add(Flatten())
model.add(Dense(100, kernel_regularizer=l2(0.001)))
model.add(Dense(50, kernel_regularizer=l2(0.001)))
model.add(Dense(10, kernel_regularizer=l2(0.001)))
model.add(Dense(1))

# compile and train the model
model.compile(optimizer='Adam', loss='mse')
model_hst = model.fit_generator(train_generator, steps_per_epoch=ceil(len(steering_angle_train)/bs),\
                                validation_data=valid_generator, \
                                validation_steps=ceil(len(steering_angle_valid)/bs), \
                                epochs=3, verbose=1)
print(model.summary())

# save and export the model
model.save('model.h5')
print('Model saved to "model.h5" file!')

# plot loss history
plt.plot(model_hst.history['loss'], label='Training Set')
plt.plot(model_hst.history['val_loss'], label='Validation Set')
plt.title('Model MSE over Epochs')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
plt.show()
