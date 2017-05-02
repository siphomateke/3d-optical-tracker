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
    def __init__(self, center):
        self.center = center


def find_fiducial_marker(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    """canny = cv2.Canny(frame, 50, 255)
    cv2.imshow("Canny", canny)
    lines = cv2.HoughLinesP(canny, 1, math.pi / 180.0, 20, None, 10, 1)
    line_mat = frame.copy()
    if lines is not None:
        for line in lines:
            cv2.line(line_mat, tuple(line[0, :2]), tuple(line[0, 2:]), (0, 0, 255), 2)
    cv2.imshow("Lines", line_mat)"""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_thresh = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    yellow_mask = cv2.dilate(yellow_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    cv2.imshow("Yellow", yellow_mask)

    yellow_regions = cv2.bitwise_and(gray, gray, mask=yellow_mask)
    cv2.imshow("Yellow regions", yellow_regions)

    harris_corners = cv2.cornerHarris(yellow_regions, 2, 3, 0.04)
    ret, harris_thresh = cv2.threshold(harris_corners, 0.02 * harris_corners.max(), 255, 0)
    harris_thresh = np.uint8(harris_thresh)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(harris_thresh)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # Now draw them
    harris_vis = np.zeros(frame.shape, np.uint8)
    harris_vis[np.int0(corners[:, 1]), np.int0(corners[:, 0])] = [0, 255, 0]
    harris_vis = cv2.dilate(harris_vis, None)
    harris_vis = cv2.add(harris_vis, frame)

    cv2.imshow('Harris corners', harris_vis)

    img_size = (frame.shape[0] + frame.shape[1]) / 2.0
    block_size = int(img_size / 180.0)
    if block_size % 2 == 0:
        block_size += 1
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, 20)
    cv2.imshow("Thresh", adaptive_thresh)

    _, contours, hierarchy = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_mat = frame.copy()

    detected_ellipses = []
    areas = []
    for i in xrange(len(contours)):
        contour = contours[i]
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if (len(approx) > 4) and (area > 10):
            ellipse = cv2.fitEllipse(contour)
            ellipse_center = np.array([ellipse[0][0], ellipse[0][1]])

            circle_center, radius = cv2.minEnclosingCircle(contour)
            circle_center = np.array([circle_center[0], circle_center[1]])

            cv2.drawContours(contour_mat, [approx], -1, (255, 0, 0), 1)
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(contour_mat, str(len(approx)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,255,255), 1)

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
                    areas.append(area)

    centers = {}
    for i in xrange(len(detected_ellipses)):
        centers[i] = np.array(detected_ellipses[i][0])

    clusters = {}
    MAX_DIST = 20
    for i in xrange(len(detected_ellipses)):
        center = centers[i]
        found_cluster = False
        for template_idx, cluster in clusters.iteritems():
            dist = np.linalg.norm(center - centers[template_idx])
            if dist < MAX_DIST:
                if areas[i] > areas[template_idx]:
                    cluster.insert(0, i)
                    clusters[i] = cluster
                else:
                    cluster.append(i)
                found_cluster = True
                break
        if not found_cluster:
            clusters[i] = []
            clusters[i].append(i)

    detected_markers = []  # type: list[Marker]
    for template_idx, cluster in clusters.iteritems():
        if len(cluster) > 1:
            ellipse = detected_ellipses[cluster[0]]
            center = np.float64([0, 0])
            for c_idx in cluster:
                center += centers[c_idx]
            center /= len(cluster)
            detected_markers.append(Marker(center))
            # cv2.ellipse(contour_mat, ellipse, (0, 0, 150), -1)

    for marker in detected_markers:
        cv2.circle(contour_mat, (int(marker.center[0]), int(marker.center[1])), 5, (0, 150, 0), -1)
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
