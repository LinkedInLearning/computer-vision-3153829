import cv2
import numpy as np

# Read the image
img_array = cv2.imread("rpi4.jpg", cv2.IMREAD_COLOR)

# Convolve with the box blur kernel
box_kernel = np.full((5, 5), 0.04)
box_result = cv2.filter2D(img_array, -1, box_kernel)

# Display convolution
cv2.imshow("Box Blur", box_result)
cv2.waitKey()
cv2.destroyAllWindows()

# Convolve with the gaussian kernel
gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16.0
gaussian_result = cv2.filter2D(img_array, -1, gaussian_kernel)

# Display convolution
cv2.imshow("Gaussian Blur", gaussian_result)
cv2.waitKey()
cv2.destroyAllWindows()

# Convolve with the sharpen kernel
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpen_result = cv2.filter2D(img_array, -1, sharpen_kernel)

# Display convolution
cv2.imshow("Sharpen", sharpen_result)
cv2.waitKey()
cv2.destroyAllWindows()