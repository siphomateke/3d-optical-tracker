import cv2
from camera_sensors import CameraSensors
from ipcamutil import Cam
import numpy as np

MIN_AREA = 5
NUM_PXS = 200

ipcamera_url = "http://192.168.8.100:8080/"
cam = Cam(ipcamera_url)
cam.start()
sensor = CameraSensors(ipcamera_url)

cam2 = Cam("http://192.168.8.112:8080/")
cam2.start()


def find_marker(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
    cv2.GaussianBlur(gray, (7, 7), 0)

    # Find brightest pixels
    _, thresh_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    max_search = cv2.bitwise_and(gray, gray, mask=thresh_mask)
    thresh = np.zeros(gray.shape, np.uint8)
    for i in range(NUM_PXS):
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(max_search)
        thresh[max_loc[1], max_loc[0]] = 255
        max_search[max_loc[1], max_loc[0]] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

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


then = cv2.getTickCount()
while True:
    now = cv2.getTickCount()
    delta = ((now - then) / cv2.getTickFrequency())
    then = now

    sensor.update()

    cam1_found_marker = False
    y, z = None, None
    if cam.is_opened():
        frame = cam.get_frame()
        cv2.imshow("Raw camera feed", frame)

        ret, xm, ym = find_marker(frame)
        contour_mat = frame.copy()
        cam1_found_marker = ret
        if ret:
            y = float(xm) / frame.shape[0]
            z = float(ym) / frame.shape[1]
            cv2.circle(contour_mat, (xm, ym), 10, (255, 0, 0), -1)
            cv2.imshow("Contour image", contour_mat)

    cam2_found_marker = False
    x, y2 = None, None
    if cam2.is_opened():
        frame = cam2.get_frame()
        cv2.imshow("Raw camera feed2", frame)

        ret, xm, ym = find_marker(frame)
        contour_mat = frame.copy()
        cam2_found_marker = ret
        if ret:
            x = float(xm) / frame.shape[0]
            y2 = float(ym) / frame.shape[1]
            cv2.circle(contour_mat, (xm, ym), 10, (255, 0, 0), -1)
            cv2.imshow("Contour image2", contour_mat)

    if cam1_found_marker and cam2_found_marker:
        y = (y2 + y) / 2.0
        loc_mat = np.zeros((600, 600, 3), np.uint8)
        cv2.circle(loc_mat, (int(x * loc_mat.shape[1]), int(y * loc_mat.shape[0])), int(-z * 100) + 100, (255, 0, 0),
                   -1)
        cv2.imshow("Estimated position", loc_mat)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cam.shut_down()
cam2.shut_down()
cv2.destroyAllWindows()
