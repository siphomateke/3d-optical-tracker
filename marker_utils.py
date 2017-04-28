import cv2
import numpy as np

MIN_AREA = 10
NUM_PXS = 200
MIN_THRESH = 200


def find_marker(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    cv2.imshow("Gray", gray)

    # Find brightest pixels
    _, thresh_mask = cv2.threshold(gray, MIN_THRESH, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thresh_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel)

    cv2.imshow("Thresh", thresh_mask)

    max_search = cv2.bitwise_and(gray, gray, mask=thresh_mask)
    cv2.imshow("Max search", max_search)
    thresh = np.zeros(gray.shape, np.uint8)
    for i in range(NUM_PXS):
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(max_search)
        thresh[max_loc[1], max_loc[0]] = 255
        max_search[max_loc[1], max_loc[0]] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Thresh2", thresh)

    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)
    if len(contours) > 0:
        areas = np.array([cv2.contourArea(c) for c in contours])
        contours = contours[np.argsort(areas)[::-1]]
        c = contours[0]
        if cv2.contourArea(c) > MIN_AREA:
            M = cv2.moments(c)
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])
            return True, x, y
    return False, None, None
