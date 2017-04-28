import cv2
from ipcamutil import Cam
from visual_odometry import PinholeCamera
import numpy as np
import marker_utils

ipcam_url = "http://192.168.1.115:8080/"
cam = Cam(ipcam_url)
cam.start()

camera_name = "LG-K8_scaled"
cam_data = PinholeCamera("camera/" + camera_name + ".npz")

chessboard_size = (9, 6)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.float32([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])


def draw(img, imgpts):
    for pt in imgpts:
        ptr = pt.ravel()
        if 0 < ptr[0] <= img.shape[1] and 0 < ptr[1] <= img.shape[0]:
            img = cv2.circle(img, (ptr[0], ptr[1]), 3, (255, 0, 0), -1)
    return img


def add_points(pt_list, pts):
    if pt_list.shape[0] > 0:
        pt_list = np.vstack((pt_list, pts))
    else:
        pt_list = pts
    return pt_list


imgp = np.array([])
pnp_solved = False
rvecs, tvecs = None, None
while True:
    if cam.is_opened():
        frame = cam.get_frame()
        undistorted = cv2.undistort(frame, cam_data.mtx, cam_data.dist)
        cv2.imshow("Cam", frame)

        img = undistorted.copy()

        ret, x, y = marker_utils.find_marker(frame)
        if ret:
            pts = np.array([[x, y]])
            cv2.circle(img, (int(x), int(y)), 20, (0, 0, 255), 5)

        if len(imgp) >= len(objp):
            # Find the rotation and translation vectors.
            if not pnp_solved:
                _, rvecs, tvecs = cv2.solvePnP(objp, imgp, cam_data.mtx, cam_data.dist)
                pnp_solved = True

            # K ^ (-1) * (x, y, 1) ^ T
            ret, x, y = marker_utils.find_marker(undistorted)
            if ret:
                R, jac = cv2.Rodrigues(rvecs)

                zp = 3  # focal length of camera
                xp = (x - cam_data.cx) * zp / cam_data.fx
                yp = (y - cam_data.cy) * zp / cam_data.fy

                pos = np.array([xp, yp, zp]).reshape(3, 1)
                T = tvecs.reshape(3, 1)

                R_inv = np.linalg.inv(R)
                projection = np.dot(R_inv, (pos - T))
                xr = projection[0]
                yr = projection[1]
                zr = projection[2]

                new_points = np.float32([[xr, yr, zr]])
                screen_points, jac = cv2.projectPoints(new_points, rvecs, tvecs, cam_data.mtx, cam_data.dist)
                backprojection_error = np.linalg.norm(np.array([screen_points.ravel()[0], screen_points.ravel()[1]]) - np.array([x, y]))
                print "Backprojection Error: {}".format(backprojection_error)
                center = (int(screen_points.ravel()[0]), int(screen_points.ravel()[1]))
                cv2.circle(img, center, 20, (0, 255, 0), 5)
        cv2.imshow("Visualization", img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('s') and ret:
        if len(imgp) < len(objp):
            print len(imgp), len(objp)
            imgp = add_points(imgp, pts)
            print imgp
            print("Saved coordinates as {}".format(objp[len(imgp) - 1]))

cam.shut_down()
cv2.destroyAllWindows()
