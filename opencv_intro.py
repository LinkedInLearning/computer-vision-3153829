import cv2 as c
import numpy as np

# Read the image

img_array = c.imread('car.jpg', c.IMREAD_COLOR)
# Remove blue content
#blue, green, red and third msut set to 0
img_array[:, :, 0] = 0

# Make the first ten rows gray
img_array[:10, :, :] = 128

# Center the image by subtracting the mean
img_array = img_array.astype(np.float64) - np.mean(img_array)

# Standarize the image by dividing the standard deviation
img_array /= np.std(img_array)

# Compute statistics
stats = f"Mean: {np.mean(img_array)}, StdDev: {np.std(img_array)}"

# Prepare the new image to be saved
img_array = c.normalize(img_array, None, 0, 255, c.NORM_MINMAX, c.CV_8U)

# Print the mean and stddev on the image
c.putText(img_array, stats, (10, 40), c.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

# Save the output image
c.imwrite('new_image.jpg', img_array)