import cv2
import numpy as np
import camutils
from camutils import CamData

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

chessboard_size = (9, 6)
chessboard_square_size = 26  # mm
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= chessboard_square_size

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

camera_name = "LG-K8"


def run(frame, save_img, calibrate):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = frame.copy()
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size,
                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_FILTER_QUADS)
    if ret:
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if save_img:
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            print "Saved image points ({} total)".format(len(imgpoints))
        cv2.drawChessboardCorners(output, chessboard_size, corners_refined, ret)

    if calibrate:
        print "Calibrating camera..."
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None,
                                                           flags=cv2.CALIB_FIX_ASPECT_RATIO)
        print "RMS : {}".format(ret)
        cam_data = CamData.create_from_data(ret, mtx, dist)
        cam_data.save("camera/" + camera_name)
        return True
    cv2.imshow("Output", output)
    return False


def main():
    cam = camutils.IPCam("http://192.168.8.100:8080/")
    cam.start()

    if cam.ready:

        save_img = False
        calibrate = False

        cv2.namedWindow("cam")
        quit_program = False
        while True:
            if cam.frame_ready:
                frame = cam.frame
                quit_program = run(frame, save_img, calibrate)
                cv2.imshow("cam", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or quit_program:
                break
            elif key & 0xFF == ord('s'):
                save_img = True
            elif key & 0xFF == ord('c'):
                calibrate = True
            else:
                calibrate = False
                save_img = False
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
