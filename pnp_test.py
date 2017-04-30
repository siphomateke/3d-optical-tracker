import cv2
import numpy as np


def draw_polygon(img, pts, color, thickness):
    if len(pts) > 0:
        for i in xrange(len(pts) - 1):
            if i == 0:
                pt1 = pts[len(pts) - 1]
                pt2 = pts[i]
                pt1 = pt1.ravel()
                pt2 = pt2.ravel()
                cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, thickness)
            pt1 = pts[i]
            pt2 = pts[i + 1]
            pt1 = pt1.ravel()
            pt2 = pt2.ravel()
            cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, thickness)

# object points needs to be float and image points need to be float

data_size = 4
imgp = np.zeros((data_size, 2), dtype=np.float32)
imgp[:, 0] = np.random.uniform(0, 500, size=data_size)
imgp[:, 1] = np.random.uniform(0, 300, size=data_size)

objp = np.zeros((data_size, 3), dtype=np.float32)
objp[:, 0] = imgp[:, 0] * np.random.uniform(-500, 500, size=1)
objp[:, 1] = imgp[:, 1] * np.random.uniform(-500, 500, size=1)

objp = objp.reshape(-1, 3)
imgp = imgp.reshape(-1, 2)

mtx = np.ones((3, 3), dtype=np.float64)
dist = np.zeros((1, 5), dtype=np.float64)

img = np.zeros((300, 500, 3), dtype=np.uint8)
if len(imgp) >= len(objp):
    _, rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, None)

    grid_points, jac = cv2.projectPoints(objp, rvecs, tvecs, mtx, None)
    draw_polygon(img, grid_points, (255, 255, 0), 6)

cv2.circle(img, tuple(imgp[0].ravel()), 5, (255, 0, 0), -1)
draw_polygon(img, imgp, (255, 0, 255), 2)
cv2.imshow("Visualization", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
