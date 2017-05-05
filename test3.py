import cv2
from camera_sensors import CameraSensors
from camutil import Cam
import numpy as np
import marker_utils

ipcam_url = "http://192.168.8.100:8080/"

cam = Cam(ipcam_url)
cam.start()


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


marker = Marker(4)
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

    cam_found_marker = False
    x, y2 = None, None
    if cam.is_opened():
        frame = cam.get_frame()[:, ::-1, :]
        if not img_size_set:
            loc_mat = np.zeros(frame.shape, np.uint8)
            img_size_set = True

        ret, xm, ym = marker_utils.find_marker(frame)
        contour_mat = frame.copy()
        cam_found_marker = ret
        if ret:
            x = xm / frame.shape[1]
            y2 = ym / frame.shape[0]
            cv2.circle(contour_mat, (int(xm), int(ym)), 10, (255, 0, 0), -1)
            # cv2.imshow("Contour image2", contour_mat)

    if cam_found_marker:
        # y = (y2 + y) / 2.0
        y = y2

        marker.set_dest(np.array([x, y, 0]))

        xd, yd = (int(marker.loc[0] * loc_mat.shape[1]), int(marker.loc[1] * loc_mat.shape[0]))
        # cv2.circle(loc_mat, (xd, yd), 4, (255, 0, 0), -1)
        if not prev_set:
            prevxd = xd
            prevyd = yd
            prev_set = True
        cv2.line(loc_mat, (xd, yd), (prevxd, prevyd), (255, 0, 0), 2)
        prevxd = xd
        prevyd = yd
        # cv2.line(loc_mat, (xd, yd), (int(xd + ((roll / 90.0) * -400)), int(yd + ((pitch / 90.0) * 400))), (0, 255, 0),3)
    else:
        timer += delta

    if cam.is_opened():
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
cv2.destroyAllWindows()
