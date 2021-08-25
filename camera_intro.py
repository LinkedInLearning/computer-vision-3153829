import picamera
import picamera.array
import cv2
import time

# Access camera and stream
camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.brightness = 60
camera.contrast = 20
camera.iso = 1600
stream = picamera.array.PiRGBArray(camera)
        
# Capture images
for i in range(10):
    camera.capture(stream, format='bgr')
    img = stream.array
    cv2.imwrite('img_{}.jpg'.format(i), img)
    stream.seek(0)
    time.sleep(0.5)
