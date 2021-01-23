import cv2
import numpy as np


class CoinExtraction:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        # self.out_path = out_path

    # Get coordinates and radius of any circle visible in the image
    def hough_transform(self):
        # Converting to gray scale
        gray_scale = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # Gaussian filtering (blurring)
        blur = cv2.GaussianBlur(gray_scale, (3, 3), 0)

        # Using Hough Transform to find circles in the image
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.5, 320, param1=200, param2=110, minRadius=60,
                                   maxRadius=400)
        circles = np.uint16(np.around(circles))
        # Check if any circles are found
        if circles is not None:
            # Convert coordinates and radius to integers
            return np.round(circles[0, :]).astype("int")

    # Get output image of marked circles on the input image and create separated coins' images
    def crop_hough(self, circles):
        cropped = []
        output = self.img.copy()
        # Crop and separate each coin
        for count, (x, y, r) in enumerate(circles, start=1):
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cropped.append(self.img[max(y - r, 0):y + r, max(x - r, 0):x + r])
            # Put a black mask over the background
            # mask = np.zeros_like(cropped, dtype=np.uint8)
            # cv2.circle(mask, (r, r), r, (255, 255, 255), -1, 8, 0)
            # out = (cropped.astype(float) * mask.astype(float)) / 255
        return output, cropped
