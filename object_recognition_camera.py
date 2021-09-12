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
ip = tflite.Interpreter(model_path='object_recognition.tflite')
ip.allocate_tensors()

# Get input and output indices
input_index = ip.get_input_details()[0]['index']
output_index = ip.get_output_details()[0]['index']

# Preprocess and analyze images
camera, stream = get_camera_stream()
for i in range(10):
    camera.capture(stream, format='jpeg')
    stream.seek(0)
    img = Image.open(stream).crop((80, 0, 560, 480))
    img = img.resize((256, 256))
    tensor = preprocess(img)

    # Send image to interpreter
    ip.set_tensor(input_index, tensor)
    ip.invoke()
    pred = ip.get_tensor(output_index)

    # Draw on image and save to file
    draw = ImageDraw.Draw(img)
    msg = 'pred: {}, prob: {}'.format(np.argmax(pred), np.max(pred))
    draw.text((10, 10), msg)
    img.save('img_{}.jpg'.format(i))
    time.sleep(0.5)

