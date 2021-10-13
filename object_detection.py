import cv2
import numpy as np
from os import listdir

# Create image window
def display(name, image):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()

# Preprocess images
def preprocess(img_dir, img_file):
    img = cv2.imread(img_dir + '/' + img_file, cv2.IMREAD_COLOR)    
    img = img.astype(np.float64) - np.mean(img)
    img /= np.std(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return img

# Create HOG detector
window_size = (32, 96)
cell_size = (8, 8)
nbins = 9
block_size = (32, 32)
block_stride = (32, 32)
hog = cv2.HOGDescriptor(window_size, block_size, block_stride, cell_size, nbins)

# Read training images
train_features = []
train_labels = []
img_dir = 'train_imgs'
for img_file in listdir(img_dir):
    img = preprocess(img_dir, img_file)
    train_features.append(hog.compute(img))
    print(train_features[0].shape)
    break
    
# Create linear SVM


# Train and save SVM


# Configure detector


# Load test images and classify

