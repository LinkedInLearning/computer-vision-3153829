import tflite_runtime.interpreter as tflite
import numpy as np
import picamera
import picamera.array
import time
import io
from PIL import Image, ImageDraw

# Create interpreter
interpreter = tflite.Interpreter(model_path='object_recognition.tflite')
interpreter.allocate_tensors()

# Get input and output indices
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Access camera and stream
camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.brightness = 60
camera.contrast = 20
camera.iso = 1600
stream = io.BytesIO()

# Preprocess and analyze images
for i in range(10):
    camera.capture(stream, format='jpeg')
    stream.seek(0)
    pil_image = Image.open(stream).crop((80, 0, 560, 480))
    pil_image = pil_image.resize((256, 256))

    # Convert to NumPy array
    test_image = np.array(pil_image)
    test_image = test_image.astype(np.float32) - np.mean(test_image)
    test_image /= np.std(test_image)

    # Send image to interpreter
    test_image = np.expand_dims(test_image, axis=0)
    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)

    # Draw on image and save to file
    draw = ImageDraw.Draw(pil_image)
    res_str = 'pred: {}, prob: {}'.format(np.argmax(pred), np.max(pred))
    draw.text((10, 10), res_str)
    pil_image.save('img_{}.jpg'.format(i))
    stream.seek(0)
    time.sleep(0.5)