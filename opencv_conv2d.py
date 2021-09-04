import cv2
import numpy as np

# Create image window
def display(name, image):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()

# Read the image
img_array = cv2.imread("rpi4.jpg", cv2.IMREAD_COLOR)
display("Original", img_array)

# Convolve with the box blur kernel
box_kernel = np.full((3, 3), 1/9)
box_result = cv2.filter2D(img_array, -1, box_kernel)
display("Box Blur", box_result)

# Convolve with the gaussian kernel
gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16.0
gaussian_result = cv2.filter2D(img_array, -1, gaussian_kernel)
display("Gaussian Blur", gaussian_result)

# Convolve with the sharpen kernel
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpen_result = cv2.filter2D(img_array, -1, sharpen_kernel)
display("Sharpen", sharpen_result)
