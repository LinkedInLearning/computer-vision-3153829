import tflite_runtime.interpreter as tflite
import numpy as np
import picamera
import time
import io
from PIL import Image, ImageDraw

# Preprocess images
def preprocess(image):
    image = np.array(image)
    image = image.astype(np.float32) - np.mean(image)
    image /= np.std(image)
    image = np.expand_dims(image, axis=0)
    return image

# Access camera and stream
def get_camera_stream():
    camera = picamera.PiCamera()
    camera.resolution = (640, 480)
    camera.brightness = 60
    camera.contrast = 20
    camera.iso = 1600
    stream = io.BytesIO()
    return camera, stream

# Create interpreter


# Get input and output indices


# Preprocess and analyze images


    # Send image to interpreter


    # Draw on image and save to file


