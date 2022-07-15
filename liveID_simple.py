# USAGE
# python live_search_simple.py 

# import the necessary packages
from imutils import paths
import numpy as np
import os
import cv2
import argparse

# 딥러닝 모델을 정의한다.
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class MiniVGGNet3:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # first CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # third CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # fourth CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
    
def get_classnames(text_filename):
    #text_filename = 'classnames.txt'
    file1 = open(text_filename, 'r')
    lines = file1.read().splitlines()
    file1.close()

    return lines

def get_prediction(img):
    # insert automatic classification routine here
    print('[INFO] Automatic search begins ...')

    # use only center part of the input image        
    height, width, channel = img.shape

    resized_img = cv2.resize(img, (image_size, image_size))
    in_data = np.asarray(resized_img)
    in_data = in_data.astype("float") / 255.0
    print("in_data shape: {}".format(in_data.shape))

    X = []
    X.append(in_data)

    X = np.array(X)
    print('\n')
    print("X.shape = {}".format(X.shape))

    # predict result
    prediction = model.predict(X)
    y = prediction.argmax()

    print('\n')
    print(prediction)

    prediction_str = label[y].lower()

    print('\n')
    print('=============================')
    print(' {}'.format(prediction_str))
    print('=============================')
    print('\n')
    print('DONE!')


# paths
datasetPath = "./dataset"
weightPath = "./weights"
uploadPath = "./upload"

# grab the list of images that we'll be describing
print("[INFO] loading images...")
label = get_classnames('classnames.txt')
for loaded_str in label:
    print(loaded_str)
print("--------- Number of classes = {} ------------".format(len(label)))

# set parameters for accuracy test
num_class = len(label)
color_depth = 3
num_epoch = 400
image_size = 256

# build model
model = MiniVGGNet3.build(width=image_size, height=image_size, depth=color_depth, classes=num_class)
model_str = 'minivggnet3'

# load weights from a matched model
dataset_name = datasetPath.split(os.path.sep)[-1]
fname = os.path.sep.join([weightPath, dataset_name + "-" + model_str + "-epoch" + str(num_epoch) + "-class" + str(num_class) + "-imageSize" + str(image_size) + "-"  + "weights-best-DA.hdf5"])

model.load_weights(fname)


# capture an image and classify it
vsource = 0
cap = cv2.VideoCapture(vsource)  # video capture source camera (Here webcam of laptop)
print('Select video source # {}'.format(vsource))

image_index = 1

# Check success
if not cap.isOpened():
    raise Exception("Could not open video device")

while(True):
    filename = 'test_' + str(image_index) + '.jpg'
    imagepath = os.path.sep.join([uploadPath, filename])
    ret, img = cap.read() # return a single frame in variable `frame`

    cv2.namedWindow('preview')
    cv2.imshow('preview', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'): # save on pressing 's' 
        cv2.imwrite(imagepath,img)
        image_index += 1
        
        get_prediction(img)
    elif key == ord('q'): #quit on pressing 'q' 
        cv2.destroyAllWindows()
        break

cap.release()
