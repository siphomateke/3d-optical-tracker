import cv2
from ipcamutil import Cam
from camera_sensors import CameraSensors
from visual_odometry import PinholeCamera
import numpy as np
import math

ipcam_url = "http://192.168.1.115:8080/"
cam = Cam(ipcam_url)
cam.start()

camera_name = "LG-K8_scaled"
cam_data = PinholeCamera("camera/" + camera_name + ".npz")

chessboard_size = (9, 6)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.float32([-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0])

def draw(img, imgpts):
    for pt in imgpts:
        ptr = pt.ravel()
        if 0 < ptr[0] <= img.shape[1] and 0 < ptr[1] <= img.shape[0]:
            img = cv2.circle(img, (ptr[0], ptr[1]), 3, (255, 0, 0), -1)
    return img


while True:
    if cam.is_opened():
        frame = cam.get_frame()
        cv2.imshow("Cam", frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = gray.copy()
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None,
                                                 cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS | cv2.CALIB_CB_CLUSTERING)

        img = frame.copy()
        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)

            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners_refined, cam_data.mtx, cam_data.dist)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cam_data.mtx, cam_data.dist)

            img = draw(img, imgpts)
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow("Visualization", img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('s'):


cam.shut_down()
cv2.destroyAllWindows()
