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


# Read training images
