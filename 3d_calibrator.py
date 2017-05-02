import cv2
from ipcamutil import Cam
from visual_odometry import PinholeCamera
import numpy as np
import marker_utils
from imutils import resize_img
import math
from matplotlib import pyplot as plt

# ipcam_url = "http://192.168.8.103:8080/"
ipcam_url = "http://192.168.1.115:8080/"
cam = Cam(ipcam_url)
cam.start()

camera_name = "LG-K8_scaled2"
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
        self.center = np.array([ellipse[0][0], ellipse[0][1]])


class MarkerFinder:
    def __init__(self):
        self.contours = None
        self.hierarchy = None

    def get_contour_children(self, idx):
        child_idx = self.hierarchy[0][idx][2]
        has_child = child_idx > -1
        children = np.int0([])
        while has_child:
            children = np.append(children, child_idx)
            child_idx = self.hierarchy[0][child_idx][2]
            has_child = child_idx > -1
        return children

    def run(self, frame):
        img_size = (frame.shape[0] + frame.shape[1]) / 2.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        processed = cv2.equalizeHist(gray)
        processed = cv2.GaussianBlur(processed, (7, 7), 1.4, 1.4)
        cv2.imshow("Processed", processed)

        canny = cv2.Canny(processed, 50, 150)
        cv2.imshow("Canny", canny)
        _, self.contours, self.hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

        contour_mat = frame.copy()

        areas = np.float32([])
        for i in xrange(len(self.contours)):
            area = cv2.contourArea(self.contours[i])
            areas = np.append(areas, area)

        detected_ellipses = []
        for i in xrange(len(self.contours)):
            contour = self.contours[i]
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = areas[i]
            children = self.get_contour_children(i)
            num_children = len(children)
            children_areas = areas[children]

            if (len(approx) > 4) and (math.pow(img_size / 2.0, 2) > area > 10) and num_children >= 3:
                # If area is bigger than all children
                if np.all(area > children_areas):
                    child_area_ratio = area / areas[children[1]]
                    if 2.1 > child_area_ratio > 1.7:
                        ellipse = cv2.fitEllipse(contour)
                        # Major and minor axes respectively
                        Ma, ma = ellipse[1]
                        aspect_ratio = float(Ma) / ma
                        if aspect_ratio < 3:
                            ellipse_center = np.array([ellipse[0][0], ellipse[0][1]])

                            circle_center, radius = cv2.minEnclosingCircle(contour)
                            circle_center = np.array([circle_center[0], circle_center[1]])

                            # Distance between approx circle center and approx ellipse center
                            center_dist = np.linalg.norm(ellipse_center - circle_center)
                            # Make sure approx ellipse is not bigger than approx circle
                            if center_dist < (Ma + ma) / 4.0 and Ma <= radius * 2.1 and ma <= radius * 2.1:
                                # Determine if contour is ellipse using difference between ellipse and actual contour
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
                                    cv2.ellipse(contour_mat, ellipse, (0, 255, 150), 1)

        detected_markers = np.array([])  # type: list[Marker]
        for ellipse in detected_ellipses:
            detected_markers = np.append(detected_markers, Marker(ellipse))

        # Sort markers
        if len(detected_markers) > 0:
            centers = np.array([marker.center for marker in detected_markers])
            detected_markers = detected_markers[np.argsort(centers[:, 0])[::-1]]  # x sort
            detected_markers = detected_markers[np.argsort(centers[:, 1])]  # y sort
            prev_center = None
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in xrange(len(detected_markers)):
                marker = detected_markers[i]
                cv2.circle(contour_mat, (int(marker.center[0]), int(marker.center[1])), 2, (0, 150, 0), -1)
                cv2.putText(contour_mat, str(i), tuple(np.int0(marker.center)), font, 1, (255, 0, 255), 2)
                if prev_center is not None:
                    cv2.line(contour_mat, tuple(np.int0(marker.center)), tuple(np.int0(prev_center)), (0, 0, 255), 2)
                prev_center = marker.center
        cv2.imshow('Objects Detected', contour_mat)


def find_fiducial_marker(frame):
    mFinder = MarkerFinder()
    mFinder.run(frame)


print cam_data.mtx.shape
print cam_data.dist.shape

imgp = np.float32([])
pnp_solved = False
rvecs, tvecs = None, None
while True:
    if cam.is_opened():
        frame = cam.get_frame()
        cv2.imshow("Cam", frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        undistorted = cv2.undistort(frame, cam_data.mtx, cam_data.dist)

        img = frame.copy()

        ret, x, y = marker_utils.find_marker(frame)
        if ret:
            pts = np.float32([[x, y]])
            cv2.circle(img, (int(x), int(y)), 20, (0, 0, 255), 2)

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
