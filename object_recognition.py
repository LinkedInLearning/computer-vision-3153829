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
ip = tflite.Interpreter(model_path='object_recognition.tflite')
ip.allocate_tensors()

# Get input/output indices
input_index = ip.get_input_details()[0]['index']
output_index = ip.get_output_details()[0]['index']

# Load test images
img_dir = 'test_imgs'
num_correct = 0
for img_file in listdir(img_dir):
    
    # Send image to interpreter
    img = preprocess(img_dir, img_file)
    ip.set_tensor(input_index, img)

    # Launch interpreter and get prediction
    ip.invoke()
    preds = ip.get_tensor(output_index)
    
    # Test classifications
    label = int(img_file.split('_')[0])
    if np.argmax(preds) == label:
        num_correct += 1

# Display accuracy
num_imgs = len(listdir(img_dir))
print('{} correct out of {}'.format(num_correct, num_imgs))