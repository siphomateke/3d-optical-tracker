import cv2
from ipcamutil import Cam
from visual_odometry import PinholeCamera
import numpy as np
import marker_utils
import math
from imutils import resize_img
import math
from matplotlib import pyplot as plt

# ipcam_url = "http://192.168.8.103:8080/"
ipcam_url = "http://192.168.1.115:8080/"
cam = Cam(ipcam_url)
cam.start()

camera_name = "LG-K8_scaled"
cam_data = PinholeCamera("camera/" + camera_name + ".npz")

objp = np.float32([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])


def draw(img, imgpts):
    for pt in imgpts:
        ptr = pt.ravel()
        if 0 < ptr[0] <= img.shape[1] and 0 < ptr[1] <= img.shape[0]:
            img = cv2.circle(img, (ptr[0], ptr[1]), 3, (255, 0, 0), -1)
    return img


def draw_polygon(img, pts, color):
    if len(pts) > 0:
        for i in xrange(len(pts) - 1):
            if i == 0:
                pt1 = pts[len(pts) - 1]
                pt2 = pts[i]
                pt1 = pt1.ravel()
                pt2 = pt2.ravel()
                cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)
            pt1 = pts[i]
            pt2 = pts[i + 1]
            pt1 = pt1.ravel()
            pt2 = pt2.ravel()
            cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)


def add_points(pt_list, pts):
    if pt_list.shape[0] > 0:
        pt_list = np.vstack((pt_list, pts))
    else:
        pt_list = pts
    return pt_list


class Marker:
    def __init__(self, ellipse):
        self.ellipse = ellipse


def find_fiducial_marker(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img_size = (frame.shape[0] + frame.shape[1]) / 2.0
    block_size = int(img_size / 180.0)
    if block_size % 2 == 0:
        block_size += 1
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size,
                                            20)
    cv2.imshow("Thresh", adaptive_thresh)

    _, contours, hierarchy = cv2.findContours(adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected_ellipses = []
    for i in xrange(len(contours)):
        contour = contours[i]
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if (len(approx) > 4) and (area > 10):
            ellipse = cv2.fitEllipse(contour)
            ellipse_center = np.array([ellipse[0][0], ellipse[0][1]])

            circle_center, radius = cv2.minEnclosingCircle(contour)
            circle_center = np.array([circle_center[0], circle_center[1]])

            center_dist = np.linalg.norm(ellipse_center - circle_center)
            # Major and minor axes respectively
            Ma, ma = ellipse[1]
            if center_dist < (Ma + ma) / 4.0 and Ma <= radius * 2.1 and ma <= radius * 2.1:
                # Determine if contour is ellipse using difference between filled fitted ellipse and actual contour
                x, y, w, h = cv2.boundingRect(contour)

                # Draw estimated ellipse mask
                ellipse_mask = np.zeros((h, w, 1), np.uint8)
                ellipse_translated = ((ellipse[0][0] - x, ellipse[0][1] - y), ellipse[1], ellipse[2])
                cv2.ellipse(ellipse_mask, ellipse_translated, (255, 255, 255), -1)

                # Draw actual contour mask
                contour_mask = np.zeros((h, w, 1), np.uint8)
                contour_translated = np.array(contour) - np.array([x, y])
                cv2.drawContours(contour_mask, [contour_translated], 0, (255, 255, 255), -1)

                # Find difference between contour and ellipse mask
                absdiff = cv2.absdiff(contour_mask, ellipse_mask)
                # Count and normalize difference
                shape_diff = cv2.countNonZero(absdiff) / float(area)

                if shape_diff < 0.1:
                    detected_ellipses.append(ellipse)

    detected_markers = []  # type: list[Marker]
    for ellipse in detected_ellipses:
        detected_markers.append(Marker(ellipse))
        cv2.partition

    contour_mat = frame.copy()
    for marker in detected_markers:
        ellipse_center = marker.ellipse[0]
        cv2.circle(contour_mat, (int(ellipse_center[0]), int(ellipse_center[1])), 2, (0, 150, 0), -1)
        cv2.ellipse(contour_mat, marker.ellipse, (0, 0, 150), 1)
    cv2.imshow('Objects Detected', contour_mat)
    # plt.imshow(cv2.cvtColor(contour_mat, cv2.COLOR_BGR2RGB))
    # plt.show()

    """sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)
    grad = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
    cv2.imshow("Sobel", grad)"""


print cam_data.mtx.shape
print cam_data.dist.shape

imgp = np.float32([])
pnp_solved = False
rvecs, tvecs = None, None
while True:
    if cam.is_opened():
        frame = cam.get_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        undistorted = cv2.undistort(frame, cam_data.mtx, cam_data.dist)
        cv2.imshow("Cam", frame)

        img = frame.copy()

        ret, x, y = marker_utils.find_marker(frame)
        if ret:
            pts = np.float32([[x, y]])
            cv2.circle(img, (int(x), int(y)), 20, (0, 0, 255), 5)

        if len(imgp) >= len(objp):
            # Find the rotation and translation vectors.
            if not pnp_solved:
                if rvecs is None:
                    print objp
                    print imgp
                    _, rvecs, tvecs = cv2.solvePnP(objp.reshape(-1, 3), imgp.reshape(-1, 2), cam_data.mtx,
                                                   cam_data.dist)
                else:
                    _, rvecs, tvecs = cv2.solvePnP(objp, imgp.reshape(-1, 2), cam_data.mtx, cam_data.dist, rvec=rvecs,
                                                   tvec=tvecs, flags=cv2.SOLVEPNP_ITERATIVE)
                    # pnp_solved = True

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

                new_points = np.float32([[xr, yr, zr], [xr, 0, zr]])
                screen_points, jac = cv2.projectPoints(new_points, rvecs, tvecs, cam_data.mtx, cam_data.dist)
                backprojection_error = np.linalg.norm(
                    np.array([screen_points[0].ravel()[0], screen_points[0].ravel()[1]]) - np.array([x, y]))
                print "Backprojection Error: {}".format(backprojection_error)
                center = (int(screen_points[0].ravel()[0]), int(screen_points[0].ravel()[1]))
                center2 = (int(screen_points[1].ravel()[0]), int(screen_points[1].ravel()[1]))
                cv2.circle(img, center, 5, (0, 255, 0), -1)
                cv2.circle(img, center2, 5, (255, 0, 0), -1)

            grid_points, jac = cv2.projectPoints(objp, rvecs, tvecs, cam_data.mtx, cam_data.dist)
            draw_polygon(img, grid_points, (255, 255, 0))
        draw_polygon(img, imgp, (255, 0, 255))
        cv2.imshow("Visualization", img)

        find_fiducial_marker(frame)

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
