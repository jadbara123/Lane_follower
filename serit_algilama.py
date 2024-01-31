import cv2
import numpy as np
import new_sim

carlo = new_sim.Carlo()
while 1:
    carlo.run()
    ahmet = carlo.yeni_alan()
    cv2.circle(ahmet, (20,24), 4, (255,0,0), 2)

    cv2.imshow("as", ahmet)
    cv2.waitKey(1)