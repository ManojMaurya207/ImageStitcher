from stitching.pipeline import stitch_images
import cv2

if __name__ == "__main__":
    img1 = cv2.imread("data/sample/1.jpg")
    img2 = cv2.imread("data/sample/2.jpg")

    result = stitch_images(img1, img2)

    cv2.imwrite("outputs/result.jpg", result)