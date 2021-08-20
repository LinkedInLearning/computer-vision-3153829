from PIL import Image
import tflite_runtime.interpreter as tflite
import numpy as np
from os import listdir

# Create interpreter
interpreter = tflite.Interpreter(model_path='object_recognition.tflite')
interpreter.allocate_tensors()

# Get input and output indices
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Load test images
img_dir = 'test_imgs'
num_correct = 0
for img_file in listdir(img_dir):
    
    # Get image and label
    img = np.array(Image.open(img_dir + '/' + img_file))
    img = img.astype(np.float32) - np.mean(img)
    img /= np.std(img)
    label = int(img_file.split('_')[0])
    
    # Send image to interpreter
    test_image = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_index, test_image)
    
    # Launch interpreter and get prediction
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    if np.argmax(predictions) == label:
        num_correct += 1

# Display accuracy
num_imgs = len(listdir(img_dir))
print('{} correct out of {}'.format(num_correct, num_imgs))
print('Accuracy: {}'.format(num_correct/num_imgs))