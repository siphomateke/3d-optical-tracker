import cv2
import numpy as np
import ipcamutil


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

chessboard_size = (9, 6)
chessboard_square_size = 26 # mm
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= chessboard_square_size

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

camera_name = "LG-K8_scaled"


def run(frame, save_img, calibrate):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = gray.copy()
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size)
    if ret:
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if save_img:
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            print "Saved image points ({} total)".format(len(imgpoints))
        cv2.drawChessboardCorners(output, chessboard_size, corners_refined, ret)
    if calibrate:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        np.savez("camera/" + camera_name, ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        print "Successfully saved {} camera parameters".format(camera_name)
    cv2.imshow("Output", output)


def main():
    cam = ipcamutil.Cam("http://192.168.8.100:8080/")
    cam.start()

    save_img = False
    calibrate = False

    cv2.namedWindow("cam")
    while True:
        if cam.is_opened():
            frame = cam.get_frame()
            run(frame, save_img, calibrate)
            cv2.imshow("cam", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):
            save_img = True
        elif key & 0xFF == ord('c'):
            calibrate = True
        else:
            calibrate = False
            save_img = False
    cam.shut_down()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
