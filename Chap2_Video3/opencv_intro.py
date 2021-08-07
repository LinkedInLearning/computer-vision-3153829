import cv2
import numpy as np

# Read the image
img_array = cv2.imread("rpi4.jpg", cv2.IMREAD_COLOR)

# Remove the blue content
img_array[:, :, 0] = 0

# Make the first ten rows gray
img_array[:10, :, :] = 128

# Center the image by subtracting the mean
img_array = img_array.astype(np.float64) - np.mean(img_array)

# Standarize the image by dividing the standard deviation
img_array /= np.std(img_array)

# Prepare the new image to be saved
img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Print the mean and stddev on the image
msg = "Mean: {}, StdDev: {}".format(np.mean(img_array), np.std(img_array))
cv2.putText(img_array, msg, (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255))

# Save the output image
cv2.imwrite("new_image.jpg", img_array)