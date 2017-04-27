import cv2
from ipcamutil import Cam
from camera_sensors import CameraSensors
from visual_odometry import PinholeCamera
import numpy as np

ipcam_url = "http://192.168.8.100:8080/"
cam = Cam(ipcam_url)
cam.start()
sensor = CameraSensors(ipcam_url)

camera_name = "LG-K8"
cam_data = PinholeCamera("camera/" + camera_name + ".npz")
# 2.6 cm
print cv2.calibrationMatrixValues(cam_data.camera_settings["mtx"], (864, 480), 1.25, 1.25)

chessboard_size = (9, 6)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


while True:
    sensor.update()
    if cam.is_opened():
        frame = cam.get_frame()
        cv2.imshow("Cam", frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = gray.copy()
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None, cv2.CALIB_CB_FAST_CHECK)

        img = frame.copy()
        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Find the rotation and translation vectors.
            ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners_refined, cam_data.mtx, cam_data.dist)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cam_data.mtx, cam_data.dist)

            img = draw(frame, corners_refined, imgpts)
        cv2.imshow("Visualization", img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cam.shut_down()
cv2.destroyAllWindows()
