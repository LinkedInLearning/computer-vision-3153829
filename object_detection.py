import cv2
import numpy as np
from os import listdir

# Configure HOG detector 
win_size = (32, 96)
block_size = (16, 16)
block_stride = (16, 16)
cell_size = (8, 8)
nbins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

# Read training images
train_features = []
train_labels = []
img_dir = "train_imgs"
for img_file in listdir(img_dir):
    img = cv2.imread(img_dir + "/" + img_file, cv2.IMREAD_COLOR)
    img = img.astype(np.float64) - np.mean(img)
    img /= np.std(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    train_features.append(hog.compute(img))
    train_labels.append(int(img_file.split('_')[0]))

# Create linear SVM
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)

# Train and save SVM
svm.train(np.array(train_features), cv2.ml.ROW_SAMPLE, np.array(train_labels))
svm.save("svm.xml")

# Configure detector
vec = svm.getSupportVectors()
rho, _, _ = svm.getDecisionFunction(0)
vec = np.append(vec, -rho)
hog.setSVMDetector(vec)

# Load test images
test_images = []
img_dir = "test_imgs"
for img_file in listdir(img_dir):
    img = cv2.imread(img_dir + "/" + img_file, cv2.IMREAD_COLOR)
    img = img.astype(np.float64) - np.mean(img)
    img /= np.std(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    test_images.append(img)

# Test detector
for test_image in test_images:
    res = hog.detectMultiScale(test_image)
    if len(res[1]) > 0 and np.max(res[1]) > 1.2:
        index = np.argmax(res[1])
        (x, y, w, h) = res[0][index]
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('result', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()