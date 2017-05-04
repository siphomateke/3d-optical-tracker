import cv2
from ipcamutil import Cam
from visual_odometry import PinholeCamera
import numpy as np
import marker_utils
from imutils import resize_img
import math
from types import *
import config.cmarker
from matplotlib import pyplot as plt

ipcam_url = "http://192.168.8.102:8080/"
# ipcam_url = "http://192.168.1.115:8080/"
cam = Cam(ipcam_url)
cam_available = cam.start()

camera_name = "LG-K8_scaled2"
cam_data = PinholeCamera("camera/" + camera_name + ".npz")

# height, width
grid_size = (4, 3)
grid_spacing = 162  # mm
x_spacing = 212
y_spacing = 144
module_y_spacing = 154
objp = np.zeros((grid_size[1] * grid_size[0], 3), np.float32)
count = 0
grid = np.float32([])
for y in xrange(grid_size[0]):
    for x in xrange(grid_size[1]):
        coord = np.array([x * x_spacing, (y * y_spacing) + (int(y / 2) * (module_y_spacing - y_spacing))])
        objp[count, :2] = coord
        count += 1
        if (x == 0 or x == grid_size[1] - 1) and (y == 0 or y == grid_size[0] - 1):
            coord3D = np.float32([coord[0], coord[1], 0])
            coord3D_up = np.float32([coord3D[0], coord3D[1], -x_spacing])
            if len(grid) == 0:
                grid = np.float32([coord3D, coord3D_up])
            else:
                grid = np.vstack((grid, coord3D))
                grid = np.vstack((grid, coord3D_up))
# objp *= grid_spacing
# objp /= x_spacing
# grid /= x_spacing

"""grid = np.float32([
    [0, 0, 0],
    [3, 0, 0],
    [0, 1 * 1.41358025, 0],
    [3, 1 * 1.41358025, 0],
    [0, 0, -2],
    [3, 0, -2],
    [0, 1 * 1.41358025, -2],
    [3, 1 * 1.41358025, -2],
])"""
print grid


# grid *= grid_spacing


def draw_points(img, pts, color, radius=3):
    for pt in pts:
        ravelled = pt.ravel()
        cv2.circle(img, (int(ravelled[0]), int(ravelled[1])), radius, color, -1)


def pt_arr_to_tuple(pt_arr):
    ravelled = pt_arr.ravel()
    return int(ravelled[0]), int(ravelled[1])


def draw_grid(img, pts, color):
    layers = np.array([pts[:4], pts[4:]])
    thickness = 2
    for i in xrange(2):
        layer = layers[i]
        pt1 = pt_arr_to_tuple(layer[0])
        pt2 = pt_arr_to_tuple(layer[1])
        cv2.line(img, pt1, pt2, color, thickness)
        pt1 = pt_arr_to_tuple(layer[1])
        pt2 = pt_arr_to_tuple(layer[3])
        cv2.line(img, pt1, pt2, color, thickness)
        pt1 = pt_arr_to_tuple(layer[3])
        pt2 = pt_arr_to_tuple(layer[2])
        cv2.line(img, pt1, pt2, color, thickness)
        pt1 = pt_arr_to_tuple(layer[2])
        pt2 = pt_arr_to_tuple(layer[0])
        cv2.line(img, pt1, pt2, color, thickness)

    layer1 = layers[0]
    layer2 = layers[1]
    pt1 = pt_arr_to_tuple(layer1[0])
    pt2 = pt_arr_to_tuple(layer2[0])
    cv2.line(img, pt1, pt2, color, thickness)
    pt1 = pt_arr_to_tuple(layer1[1])
    pt2 = pt_arr_to_tuple(layer2[1])
    cv2.line(img, pt1, pt2, color, thickness)
    pt1 = pt_arr_to_tuple(layer1[2])
    pt2 = pt_arr_to_tuple(layer2[2])
    cv2.line(img, pt1, pt2, color, thickness)
    pt1 = pt_arr_to_tuple(layer1[3])
    pt2 = pt_arr_to_tuple(layer2[3])
    cv2.line(img, pt1, pt2, color, thickness)


def rotate_points(pts, angle):
    # 2D Rotation Matrix
    R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    # Rotate points
    return np.float32(np.dot(pts.reshape(-1, 2), R))


class Marker:
    def __init__(self, ellipse, area, num_children):
        self.ellipse = ellipse
        self.center = np.array([ellipse[0][0], ellipse[0][1]])
        self.area = area
        self.num_children = num_children

    def dist_to(self, point):
        return np.linalg.norm(self.center - point)


class MarkerFinder:
    def __init__(self):
        self.image = None
        self.img_size = None
        self.gray = None
        self.contours = None
        self.hierarchy = None
        self.markers = None

    def get_contour_children(self, idx):
        """
        Searches the contour hierarchy for the indices of all children of the given contour index
        :param idx: The id of the contour to get children of
        :return: Integer Numpy Nx1 array of children indices
        @type idx: int
        """
        assert self.hierarchy is not None, "contour hierarchy is NoneType"
        assert len(self.hierarchy) > 0, "contour hierarchy empty"
        assert type(idx) is IntType, "contour idx is not an integer: %r" % idx
        assert idx in self.hierarchy[0], "contour idx not found in hierarchy: %r" % idx
        child_idx = self.hierarchy[0][idx][2]
        has_child = child_idx > -1
        children = np.int0([])
        while has_child:
            children = np.append(children, child_idx)
            child_idx = self.hierarchy[0][child_idx][2]
            has_child = child_idx > -1
        return children

    def process_image(self, gray):
        """
        Processes a grayscale image to increase contrast and remove noise
        :param gray: The image to be processed
        :return: The processed image
        """
        clahe = cv2.createCLAHE(clipLimit=config.cmarker.CLAHE_CLIP_LIMIT, tileGridSize=(
            int(self.img_size / config.cmarker.CLAHE_TILE_GRID_SIZE_RATIO),
            int(self.img_size / config.cmarker.CLAHE_TILE_GRID_SIZE_RATIO)))
        processed = clahe.apply(gray)
        processed = cv2.GaussianBlur(processed, config.cmarker.BLUR_KSIZE, config.cmarker.BLUR_SIGMA,
                                     config.cmarker.BLUR_SIGMA)
        return processed

    @staticmethod
    def find_contours(image):
        """
        Finds contours in a grayscale image using canny edge detector
        :param image: The image to find contours in
        :return: Tuple : im2, contours, hierarchy
        """
        canny = cv2.Canny(image, config.cmarker.CANNY_LOWER, config.cmarker.CANNY_UPPER)
        return cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    @staticmethod
    def sort_markers(markers):
        """
        Sorts markers from top-left to bottom-right
        :param markers: The marers to sort
        :return: Numpy array of sorted markers
        """
        if len(markers) == (grid_size[0] * grid_size[1]):
            centers = np.array([marker.center for marker in markers])

            # Determine orientation of points
            int_centers = np.int0(centers.reshape(-1, 1, 2))
            approx_rect = cv2.minAreaRect(int_centers)
            angle = approx_rect[2] * (math.pi / 180)

            # Rotate points about origin
            rotated_points = rotate_points(centers, angle)

            # Approximate bounding rect to find out if shape was rotated correctly
            approx_rotated_rect = cv2.boundingRect(rotated_points)
            # If grid width is greater than height and rect width is greater than height
            wrong_rotation = False
            if not (grid_size[1] > grid_size[0] and approx_rotated_rect[2] > approx_rotated_rect[3]):
                # Rotated incorrectly, correct it
                rotated_points = rotate_points(centers, angle - (math.pi / 2))
                wrong_rotation = True

            # Find minimum x and y of rotated points
            min = np.float32([np.min(rotated_points[:, 0]), np.min(rotated_points[:, 1])])
            max = np.float32([np.max(rotated_points[:, 0]), np.max(rotated_points[:, 1])])

            # Normalize rotated points
            rotated_points -= min
            rotated_points /= (max - min)

            # Sort from top-left to bottom-right using x + (y * q), where q is larger than x can ever be
            # in this case q = max grid size * 2
            sort = np.argsort(rotated_points[:, 0] + (rotated_points[:, 1] * (np.max(np.array(grid_size)) * 2)))
            if wrong_rotation:
                sort = sort[::-1]

            markers = markers[sort]
            successful = True
        else:
            successful = False
        return successful, markers

    def find(self, image):
        """
        Finds elliptical markers in an image. Sample markers can be found in 'img/circle.jpg'
        :param image: The image to find markers in
        :return: Tuple: Whether any markers were found, Numpy array containing detected Marker objects
        """
        self.img_size = (image.shape[0] + image.shape[1]) / 2.0
        assert self.img_size > 0, "image size is <= 0"
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        processed = self.process_image(self.gray)
        cv2.imshow("Processed", processed)

        _, self.contours, self.hierarchy = self.find_contours(processed)
        self.markers = np.array([])  # type: list[Marker]

        found = False
        if len(self.contours) > 1:
            # region Area pre-calculation
            areas = np.float32([])
            for i in xrange(len(self.contours)):
                area = cv2.contourArea(self.contours[i])
                areas = np.append(areas, area)
            # endregion

            # region Contour filtering
            for i in xrange(len(self.contours)):
                contour = self.contours[i]
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                area = areas[i]
                children = self.get_contour_children(i)
                num_children = len(children)
                children_areas = areas[children]

                # num of contours must be greater than 5 for fitEllipse to work
                if len(contour) > 5 and (len(approx) > 4) and \
                        (math.pow(self.img_size / 2.0, 2) > area > config.cmarker.CONTOUR_MIN_AREA) and \
                        num_children >= 2 and np.all(area > children_areas):
                    # Check ratio to children
                    ratio_correct = False
                    for child in children:
                        if areas[child] <= 0:
                            continue
                        child_area_ratio = area / areas[child]
                        if config.cmarker.CONTOUR_MAX_CHILD_AREA_RATIO > child_area_ratio > config.cmarker.CONTOUR_MIN_CHILD_AREA_RATIO:
                            ratio_correct = True
                            break
                    # If ratio to children matches model
                    if ratio_correct:
                        ellipse = cv2.fitEllipse(contour)
                        # Major and minor ellipse axes respectively
                        major, minor = ellipse[1]
                        aspect_ratio = float(major) / minor
                        if aspect_ratio < config.cmarker.CONTOUR_MAX_ASPECT_RATIO:
                            ellipse_center = np.array([ellipse[0][0], ellipse[0][1]])

                            circle_center, radius = cv2.minEnclosingCircle(contour)
                            circle_center = np.array([circle_center[0], circle_center[1]])

                            # Distance between approx circle center and approx ellipse center
                            center_dist = np.linalg.norm(ellipse_center - circle_center)
                            # Make sure approx ellipse is not bigger than approx circle
                            if center_dist < (major + minor) / config.cmarker.CONTOUR_MAX_CENTER_DIST_RATIO and \
                                    major <= radius * config.cmarker.CONTOUR_MAJOR2RADIUS_RATIO and \
                                    minor <= radius * config.cmarker.CONTOUR_MINOR2RADIUS_RATIO:
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

                                # Find absolute difference between contour and ellipse mask
                                abs_diff = cv2.absdiff(contour_mask, ellipse_mask)
                                # Count and normalize difference
                                shape_diff = cv2.countNonZero(abs_diff) / float(area)

                                if shape_diff < config.cmarker.CONTOUR_ELLIPSE_CONTOUR_DIFF:
                                    m = Marker(ellipse, area, num_children)

                                    # region Check if there is a better marker nearby and delete worse ones
                                    found_duplicate = False
                                    markers_to_remove = []
                                    for j in xrange(len(self.markers)):
                                        m2 = self.markers[j]
                                        if m.dist_to(m2.center) < config.cmarker.MAX_MARKER_DIST:
                                            # If found a better one forget about me
                                            if m2.area > m.area and m2.num_children > m.num_children:
                                                found_duplicate = True
                                            # If existing ones are worse delete them
                                            else:
                                                markers_to_remove.append(j)

                                    for idx in markers_to_remove:
                                        np.delete(self.markers, idx)
                                    # endregion

                                    # If this is the best marker in the vicinity use this one
                                    if not found_duplicate:
                                        self.markers = np.append(self.markers, m)
            # endregion

            # Sort markers from top-left to bottom-right
            found, self.markers = self.sort_markers(self.markers)

            # region Draw found markers, debug only
            contour_mat = frame.copy()
            prev_center = None
            r = 10
            colors = [
                (0, 0, 255),
                (0, 128, 255),
                (0, 200, 200),
                (0, 255, 0),
                (200, 200, 0),
                (255, 0, 0),
                (255, 0, 255)
            ]
            for i in xrange(len(self.markers)):
                marker = self.markers[i]
                center = tuple(np.int0(marker.center))
                if prev_center is not None and found:
                    cv2.line(contour_mat, center, prev_center, (0, 0, 0), 1)
                if found:
                    color = colors[int(i / grid_size[1]) % len(colors)]
                    cv2.line(contour_mat, (center[0] - r, center[1] - r), (center[0] + r, center[1] + r), color, 1)
                    cv2.line(contour_mat, (center[0] - r, center[1] + r), (center[0] + r, center[1] - r), color, 1)
                    cv2.circle(contour_mat, center, r + 2, color, 1)
                else:
                    color = (0, 0, 255)
                    cv2.line(contour_mat, (center[0] - r, center[1] - r), (center[0] + r, center[1] + r), color, 1)
                    cv2.line(contour_mat, (center[0] - r, center[1] + r), (center[0] + r, center[1] - r), color, 1)
                    cv2.circle(contour_mat, center, r + 2, color, 1)
                # cv2.ellipse(contour_mat, marker.ellipse, color, -1)
                """if found:
                    cv2.putText(contour_mat, str(i), (center[0] - 5, center[1] - int(r*1.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)"""
                prev_center = center
            cv2.imshow('Objects Detected', contour_mat)
            # endregion

            if not found:
                self.markers = np.array([])

        if self.markers.shape[0] == 0:
            found = False

        return found, self.markers


rvec, tvec = None, None
# main_img = cv2.imread("img\\Cam.png")
marker_finder = MarkerFinder()
if cam_available:
    while True:
        if cam.is_opened():
            frame = cam.get_frame()
            # frame = main_img
            cv2.imshow("Cam", frame)

            undistorted = cv2.undistort(frame, cam_data.mtx, cam_data.dist)

            img = undistorted.copy()
            found_markers, detected_markers = marker_finder.find(frame)
            if found_markers:
                imgp = np.float32([marker.center for marker in detected_markers])
            else:
                imgp = np.float32([])

            if len(imgp) == len(objp):
                # TODO: Add check to see if solvePnP was already evaluated
                # Find the rotation and translation vectors.
                ret, rvec, tvec = cv2.solvePnP(objp.reshape(-1, 3), imgp.reshape(-1, 1, 2), cam_data.mtx,
                                               cam_data.dist, flags=cv2.SOLVEPNP_ITERATIVE)

                ret, x, y = marker_utils.find_marker(undistorted)
                # x = 550
                # y = 400
                if x is not None and y is not None:
                    K = cam_data.mtx
                    R, jac = cv2.Rodrigues(rvec)

                    Z = 0

                    R_inv = np.matrix(np.linalg.inv(R))
                    K_inv = np.matrix(np.linalg.inv(K))
                    p = np.matrix(np.array([x, y, 1]).reshape(3, 1))
                    tvec_m = np.matrix(tvec)

                    tempMat = R_inv * K_inv * p
                    tempMat2 = R_inv * tvec_m
                    s = Z + tempMat2[2, 0]
                    s /= tempMat[2, 0]
                    P = R_inv * (s * K_inv * p - tvec)
                    print "P: {}".format(np.round(P, 2) / 1000)

                    xr = P[0]
                    yr = P[1]

                    new_points = np.float32([[xr, yr, Z]])
                    screen_points, jac = cv2.projectPoints(new_points, rvec, tvec, cam_data.mtx, distCoeffs=None)
                    backprojection_error = np.linalg.norm(
                        np.array([screen_points[0].ravel()[0], screen_points[0].ravel()[1]]) - np.array([x, y]))
                    print "Backprojection Error: {}".format(backprojection_error)
                    center = (int(screen_points[0].ravel()[0]), int(screen_points[0].ravel()[1]))
                    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.circle(img, center, 5, (255, 0, 0), -1)

                grid_points2, jac = cv2.projectPoints(grid, rvec, tvec, cam_data.mtx, distCoeffs=None)
                if np.all(abs(grid_points2) < 1e6):
                    draw_grid(img, grid_points2, (0, 255, 0))
            cv2.imshow("Visualization", img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

cam.shut_down()
cv2.destroyAllWindows()
