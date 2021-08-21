import picamera
import picamera.array
import cv2

# Access camera and stream
with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
   
        # Capture images
        for i in range(10):
            camera.capture(stream, format='bgr')
            img = stream.array
            cv2.imwrite('img_{}.jpg'.format(i), img)
            stream.seek(0)