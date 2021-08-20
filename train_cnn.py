import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.preprocessing import image
from os import listdir

# Load training data
imgs = []
labels = []
num_classes = 4
img_dir = 'train_imgs'
for img_file in listdir(img_dir):
    img = image.img_to_array(image.load_img(img_dir + '/' + img_file))
    img = img.astype(np.float64) - np.mean(img)
    img /= np.std(img)
    imgs.append(img)
    label = int(img_file.split('_')[0])
    labels.append(np.eye(num_classes)[label])

# Create model
model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=imgs[0].shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(num_classes, activation='softmax'))

# Set training parameters
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(np.asarray(imgs), np.asarray(labels), epochs=10)

# Save the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
open('object_recognition.tflite', 'wb').write(converter.convert())