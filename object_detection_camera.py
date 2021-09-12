import cv2
import numpy as np
import picamera
import picamera.array
import time

# Preprocess image
def preprocess(img, size):
    img = cv2.resize(img, size)
    img = img.astype(np.float64) - np.mean(img)
    img /= np.std(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return img

# Create HOG descriptor
def get_descriptor():
    win_size = (32, 96)
    cell_size = (8, 8)
    nbins = 9
    block_size = (16, 16)
    block_stride = (16, 16)
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    return hog

# Access camera and stream
def get_camera_stream():
    camera = picamera.PiCamera()
    camera.resolution = (640, 480)
    camera.brightness = 60
    camera.contrast = 20
    camera.iso = 1600
    stream = picamera.array.PiRGBArray(camera)
    return camera, stream

# Configure HOG descriptor with SVM
hog = get_descriptor()
svm = cv2.ml.SVM_load('svm.xml')
vec = svm.getSupportVectors()
rho, _, _ = svm.getDecisionFunction(0)
vec = np.append(vec, -rho)
hog.setSVMDetector(vec)

# Preprocess and analyze images
camera, stream = get_camera_stream()
for i in range(10):
    camera.capture(stream, format='bgr')
    test_image = preprocess(stream.array, (213, 160))

    # Perform object detection and save result
    res = hog.detectMultiScale(test_image)
    if len(res[1]) > 0 and np.max(res[1]) > 1.2:
        index = np.argmax(res[1])
        (x, y, w, h) = res[0][index]
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imwrite('img_{}.jpg'.format(i), test_image)
    stream.seek(0)
    time.sleep(0.5)

