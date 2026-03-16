import numpy as np
import cv2
import glob

# ------------------------------
# Load images
# ------------------------------
image_paths = glob.glob('SampleImg/TestingFolder/*.jpeg')
images = []

for image in image_paths:
    img = cv2.imread(image)
    if img is not None:
        images.append(img)

# Stitch images
imageStitcher = cv2.Stitcher_create()
error, stitchedImg = imageStitcher.stitch(images)

if error:
    print("Image could not be stitched!")
else:
    stitchedImg = cv2.copyMakeBorder(stitchedImg, 100, 100, 100, 100, cv2.BORDER_CONSTANT, (0, 0, 0))
    cv2.imshow("Stitched Image", stitchedImg)
    cv2.waitKey(0)

    # # Convert to binary mask
    # gray = cv2.cvtColor(stitchedImg, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # # Find largest contour
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # largest_contour = max(contours, key=cv2.contourArea)

    # # Create a mask from the largest contour (filled white)
    # filled_mask = np.zeros_like(thresh, dtype=np.uint8)
    # cv2.drawContours(filled_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # cv2.imshow("Filled Largest Contour Mask", filled_mask)
    # cv2.waitKey(0)

    # # Convert mask to 0/1 for maximal-inscribed-rectangle
    # mask_bin = (filled_mask > 0).astype(np.uint8)

    # # Maximal-inscribed-rectangle algorithm
    # def maximal_inscribed_rectangle(binary_mask):
    #     h, w = binary_mask.shape
    #     heights = [0] * w
    #     max_area = 0
    #     max_rect = (0, 0, 0, 0)  # x, y, width, height

    #     for row in range(h):
    #         for col in range(w):
    #             if binary_mask[row, col] == 1:
    #                 heights[col] += 1
    #             else:
    #                 heights[col] = 0

    #         stack = []
    #         for i in range(w + 1):
    #             cur_height = heights[i] if i < w else 0
    #             while stack and cur_height < heights[stack[-1]]:
    #                 h_rect = heights[stack.pop()]
    #                 left = stack[-1] + 1 if stack else 0
    #                 width = i - left
    #                 area = h_rect * width

    #                 if area > max_area:
    #                     max_area = area
    #                     x = left
    #                     y = row - h_rect + 1
    #                     max_rect = (x, y, width, h_rect)

    #             stack.append(i)
    #     return max_rect

    # x, y, w, h = maximal_inscribed_rectangle(mask_bin)

    # # Crop stitched image
    # cropped = stitchedImg[y:y+h, x:x+w]
    # cv2.imshow("Cropped Maximal Rectangle (Contour-Based)", cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



















# -----------------------------------Sample--------------------------------------


# import numpy as np
# import cv2
# import glob

# image_path = glob.glob('Slides/*.JPG')
# images =[]

# for idx, image in enumerate(image_path):
#     img = cv2.imread(image)
#     images.append(img)
#     # cv2.imshow(f"Image {idx}", img)
#     # cv2.waitKey(0)

# imageStitcher = cv2.Stitcher_create()

# error,stichedImg = imageStitcher.stitch(images)

# if not error:
#     cv2.imwrite("stitchedOutput.png", stichedImg)
#     cv2.imshow("Stitched Image", stichedImg)
#     cv2.waitKey(0)


# # ------------------------------
# # 1. Convert stitched image to grayscale
# # ------------------------------
# gray = cv2.cvtColor(stichedImg, cv2.COLOR_BGR2GRAY)

# # ------------------------------
# # 2. Threshold to get binary mask
# # ------------------------------
# _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# # ------------------------------
# # 3. Find contours
# # ------------------------------
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # ------------------------------
# # 4. Pick the largest contour
# # ------------------------------
# largest = max(contours, key=cv2.contourArea)

# # Draw contour for debugging
# debug_contour = stichedImg.copy()
# cv2.drawContours(debug_contour, [largest], -1, (0, 0, 255), 5)
# cv2.imshow("Largest Contour", debug_contour)
# cv2.waitKey(0)

# # # ------------------------------
# # # 5. Crop using bounding box
# # # ------------------------------
# # x, y, w, h = cv2.boundingRect(largest)
# # cropped = stichedImg[y:y+h, x:x+w]

# # cv2.imshow("Cropped Output", cropped)
# # cv2.imwrite("stitched_cropped.png", cropped)
# # cv2.waitKey(0)


# # Get bounding rectangle of the largest contour
# x, y, w, h = cv2.boundingRect(largest)

# # Draw rectangle on a copy of the stitched image
# output_img = stichedImg.copy()
# cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 5)  # Green rectangle with thickness 5

# # Show the image
# cv2.imshow("Largest Bounding Rectangle", output_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
