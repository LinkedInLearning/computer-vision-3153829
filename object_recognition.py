from PIL import Image
import tflite_runtime.ip as tflite
import numpy as np
from os import listdir

# Preprocess images
def preprocess(img_dir, img_file):
    img = np.array(Image.open(img_dir + '/' + img_file))
    img = img.astype(np.float32) - np.mean(img)
    img /= np.std(img)
    img = np.expand_dims(img, axis=0)
    return img

# Create interpreter


# Get input/output indices


# Load test images

    
    # Send image to interpreter


    # Launch interpreter and get prediction

    
    # Test classifications


# Display accuracy
