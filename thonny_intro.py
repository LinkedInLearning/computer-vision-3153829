# Import packages
import cv2
import numpy as np

# Create array
img_array = np.full((400, 600), 0.5)

# Display array as image
cv2.imshow("Window", img_array)

# Wait for a keypress
cv2.waitKey()
cv2.destroyAllWindows()