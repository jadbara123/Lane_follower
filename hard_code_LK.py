import cv2
import numpy as np

class lane_keep:
    def __init__(self):
        pass

    def main(self, image_right, image_left):
        self.image_right = image_right
        self.image_left = image_left
        crx = np.count_nonzero(image_right==255)
        clx = np.count_nonzero(image_left==255)
        print(crx)
        if crx < 200 or clx > 300:
            return 3
        elif clx < 200 or crx > 300:
            return 4
        else:
            return 2
        
    def show_me_the_meaning(self):
        cv2.imshow(self.image_right)
        
    def find_largest_contour_center(self, binary_image):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)

        # Calculate the centroid of the largest contour
        cx = int(moments['m10'] / (moments['m00']+0.000000000000000004))
        cy = int(moments['m01'] / (moments['m00']+0.000000000000000004))

        return cx, cy
        
