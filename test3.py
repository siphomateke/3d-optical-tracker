import cv2
from camera_sensors import CameraSensors
from ipcamutil import Cam
import numpy as np
from imutils import rotate_img
from imutils import resize_img

MIN_AREA = 10
NUM_PXS = 200

# ipcamera_url = "http://192.168.8.100:8080/"
ipcam1_url = "http://192.168.8.112:8080/"
ipcam2_url = "http://192.168.8.100:8080/"

cam = Cam(ipcam1_url)
cam.start()
sensor = CameraSensors(ipcam1_url)

cam2 = Cam(ipcam2_url)
cam2.start()


def find_marker(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
    cv2.GaussianBlur(gray, (13, 13), 0)

    # Find brightest pixels
    _, thresh_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

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


class Marker:
    def __init__(self, smoothing):
        self.loc = np.float64([0, 0, 0])
        self.loc_set = False
        self.dest = np.float64([0, 0, 0])
        self.smoothing = smoothing

    def set_dest(self, dest):
        self.dest = dest
        if not self.loc_set:
            self.loc = self.dest
            self.loc_set = True

    def update(self):
        if self.smoothing < 1:
            self.smoothing = 1
        self.loc += (self.dest - self.loc) / self.smoothing

marker = Marker(1)
loc_mat = np.zeros((640, 1200, 3), np.uint8)
img_size_set = False
prevxd, prevyd = None, None
prev_set = False

then = cv2.getTickCount()
timer = 0
while True:
    now = cv2.getTickCount()
    delta = ((now - then) / cv2.getTickFrequency())
    then = now

    sensor.update()

    cam1_found_marker = False
    y, z = None, None
    if cam.is_opened():
        frame = cam.get_frame()
        frame = rotate_img(frame, -sensor.get_sensor("rotation").value + 90)
        # cv2.imshow("Raw camera feed", frame)

        ret, xm, ym = find_marker(frame)
        contour_mat = frame.copy()
        cam1_found_marker = ret
        if ret:
            y = float(xm) / frame.shape[1]
            z = float(ym) / frame.shape[0]
            cv2.circle(contour_mat, (xm, ym), 10, (255, 0, 0), -1)
            # cv2.imshow("Contour image", contour_mat)

    cam2_found_marker = False
    x, y2 = None, None
    if cam2.is_opened():
        frame = cam2.get_frame()[:, ::-1, :]
        if not img_size_set:
            loc_mat = np.zeros(frame.shape, np.uint8)
            img_size_set = True
        # cv2.imshow("Raw camera feed2", frame)

        ret, xm, ym = find_marker(frame)
        contour_mat = frame.copy()
        cam2_found_marker = ret
        if ret:
            x = float(xm) / frame.shape[1]
            y2 = float(ym) / frame.shape[0]
            cv2.circle(contour_mat, (xm, ym), 10, (255, 0, 0), -1)
            # cv2.imshow("Contour image2", contour_mat)

    if cam2_found_marker:
        # y = (y2 + y) / 2.0
        y = y2

        marker.set_dest(np.array([x, y, 0]))

        xd, yd = (int(marker.loc[0] * loc_mat.shape[1]), int(marker.loc[1] * loc_mat.shape[0]))
        # cv2.circle(loc_mat, (xd, yd), 4, (255, 0, 0), -1)
        if not prev_set:
            prevxd = xd
            prevyd = yd
            prev_set = True
        roll = sensor.get_sensor("roll").value
        pitch = sensor.get_sensor("pitch").value
        thickness = int((-roll / 90) * 10)
        if thickness < 1:
            thickness = 1
        cv2.line(loc_mat, (xd, yd), (prevxd, prevyd), (255, 0, 0), thickness=thickness)
        prevxd = xd
        prevyd = yd
        #cv2.line(loc_mat, (xd, yd), (int(xd + ((roll / 90.0) * -400)), int(yd + ((pitch / 90.0) * 400))), (0, 255, 0),3)
    else:
        timer += delta

    cv2.imshow("Estimated position", cv2.addWeighted(loc_mat, 0.6, frame, 0.4, 0))
    # cv2.imshow("Estimated position", resize_img(loc_mat, height=640))

    if timer > 1:
        prev_set = False
        marker.loc_set = False
        timer = 0

    marker.update()

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('c'):
        loc_mat = np.zeros(frame.shape, np.uint8)

cam.shut_down()
cam2.shut_down()
cv2.destroyAllWindows()
