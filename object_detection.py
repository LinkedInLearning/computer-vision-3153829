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
    print(train_features[0].shape)
    break
    
# Create linear SVM

# Train and save SVM

# Configure detector

# Load test images

# Test detector