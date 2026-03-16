import numpy as np
import cv2
import glob

# Load images
image_paths = glob.glob('PanoImg/*.jpg')
images = []
for image in image_paths:
    img = cv2.imread(image)
    if img is not None:
        images.append(img)

cv2.imshow("Sample",images[0])
cv2.waitKey(0)

