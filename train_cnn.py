import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.preprocessing import image
from os import listdir

# Preprocess images
def preprocess(img_dir, img_file):
    img = image.img_to_array(image.load_img(img_dir + '/' + img_file))    
    img = img.astype(np.float64) - np.mean(img)
    img /= np.std(img)
    return img

# Load training data
def load_data(img_dir, num_classes):
    for img_file in listdir(img_dir):
        imgs = []
        labels = []
        img = preprocess(img_dir, img_file)
        imgs.append(img)
        label = int(img_file.split('_')[0])
        labels.append(np.eye(num_classes)[label])
        return imgs, labels

# Initialize the model


# Compile the model


# Train the model


# Save the model in TensorFlow Lite format