import cv2
import numpy as np
import camutils
from visual_odometry import PinholeCamera
from matplotlib import pyplot as plt

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

chessboard_size = (8, 5)
chessboard_square_size = 48  # mm
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= chessboard_square_size

gridp = np.float32([])
for x in xrange(chessboard_size[0]):
    for y in xrange(chessboard_size[1]):
        for z in xrange(5):
            point = np.float32([x, y, -z]) * chessboard_square_size
            if gridp.shape[0] > 0:
                gridp = np.vstack((gridp, point))
            else:
                gridp = point

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

camera_name = "LG-K8_scaled2"
cam_data = PinholeCamera("camera/" + camera_name + ".npz")


def draw_grid(img, pts, color):
    if len(pts) > 0:
        """for i in xrange(len(pts) - 1):
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
            cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)"""
        for pt in pts:
            ravelled = pt.ravel()
            x = ravelled[0]
            y = ravelled[1]
            if abs(x) < 1e6 and abs(y) < 1e6:
                cv2.circle(img, (int(x), int(y)), 2, color, -1)


def run(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed = gray.copy()

    cv2.imshow("Processed", processed)
    ret, corners = cv2.findChessboardCorners(processed, chessboard_size, flags=cv2.CALIB_CB_FILTER_QUADS)
    # create mat to draw results
    output = frame.copy()
    cv2.drawChessboardCorners(output, chessboard_size, corners, ret)
    if ret:
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        _, rvecs, tvecs = cv2.solvePnP(objp.reshape(-1, 3), corners_refined.reshape(-1, 2), cam_data.mtx, cam_data.dist)

        grid_points, jac = cv2.projectPoints(gridp, rvecs, tvecs, cam_data.mtx, cam_data.dist)
        draw_grid(output, grid_points, (255, 255, 0))
    cv2.imshow("Output", output)

    """processed = gray.copy()
    cv2.imshow("Processed", processed)
    processed = np.float32(processed)

    # find Harris corners
    harris_corners = cv2.cornerHarris(processed, 2, 3, 0.04)
    normalized = harris_corners.astype("float")
    normalized /= (normalized.sum())
    normalized *= 255

    cv2.imshow("Harris2", normalized)
    harris_corners = cv2.dilate(harris_corners, None)
    ret, harris_corners = cv2.threshold(harris_corners, 0.05 * harris_corners.max(), 255, 0)
    harris_corners = np.uint8(harris_corners)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(harris_corners)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    res = np.int0(corners)
    display = frame.copy()
    display[res[:, 1], res[:, 0]] = [0, 255, 0]

    cv2.imshow("Harris", display)"""


def main():
    cam = camutils.Cam("http://192.168.8.103:8080/")
    cam.start()

    cv2.namedWindow("cam")
    while True:
        if cam.is_opened():
            frame = cam.get_frame()
            run(frame)
            cv2.imshow("cam", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    cam.shut_down()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
