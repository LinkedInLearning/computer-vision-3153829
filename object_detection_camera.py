import cv2
import numpy as np
import picamera
import picamera.array
import time

# Create HOG descriptor
win_size = (32, 96)
cell_size = (8, 8)
nbins = 9
block_size = (16, 16)
block_stride = (16, 16)
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

# Load SVM and configure descriptor
svm = cv2.ml.SVM_load('svm.xml')
vec = svm.getSupportVectors()
rho, _, _ = svm.getDecisionFunction(0)
vec = np.append(vec, -rho)
hog.setSVMDetector(vec)

# Access camera and stream
camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.brightness = 60
camera.contrast = 20
camera.iso = 1600
stream = picamera.array.PiRGBArray(camera)

# Preprocess and analyze images
for i in range(10):
    camera.capture(stream, format='bgr')
    test_image = cv2.resize(stream.array, (213, 160))
    test_image = test_image.astype(np.float64) - np.mean(test_image)
    test_image /= np.std(test_image)
    test_image = cv2.normalize(test_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Perform object detection and save result
    res = hog.detectMultiScale(test_image)
    if len(res[1]) > 0 and np.max(res[1]) > 1.2:
        index = np.argmax(res[1])
        (x, y, w, h) = res[0][index]
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imwrite('img_{}.jpg'.format(i), test_image)
    stream.seek(0)
    time.sleep(0.5)