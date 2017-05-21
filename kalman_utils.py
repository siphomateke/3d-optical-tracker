import cv2
import numpy as np
from camutils import IPCam
from marker_utils import MarkerFinder

cv2.namedWindow("kalman")
dynamParams = 4
measureParams = 2
kf = cv2.KalmanFilter(dynamParams, measureParams)
# state matrix A
kf.transitionMatrix = np.eye(dynamParams, dtype=np.float32)
# kf.transitionMatrix[0, 2] = dT
# kf.transitionMatrix[1, 3] = dT
# measurement matrix H
kf.measurementMatrix = np.zeros((measureParams, dynamParams), dtype=np.float32)
kf.measurementMatrix[0, 0] = 1
kf.measurementMatrix[1, 1] = 1
# process noise covariance matrix, Q
kf.processNoiseCov = np.eye(dynamParams, dtype=np.float32) * 1e-2
kf.processNoiseCov[2, 2] = 1.0
kf.processNoiseCov[3, 3] = 1.0
# measurement noise R
kf.measurementNoiseCov = np.eye(measureParams, dtype=np.float32) * 1e-2
kf.errorCovPre = np.eye(dynamParams, dtype=np.float32)

marker_finder = MarkerFinder()
cam = IPCam("http://192.168.8.101:8080/", "", "IPCam")
cam_started = cam.start()
not_found_count = 0
then = -1
now = -1

state = np.zeros((4, 1), np.float32)

found_before = False
if cam_started:
    while True:
        if cam.frame_ready:
            now = cv2.getTickCount()
            if then == -1:
                then = now
            time = (now - then) / cv2.getTickFrequency()  # seconds
            then = now

            frame = cam.frame
            found_markers, markers = marker_finder.find_markers(frame)

            """if found_before:
                kf.transitionMatrix[0, 2] = time
                kf.transitionMatrix[1, 3] = time
                state = kf.predict()

            if found_markers:
                if not found_before:
                    state = np.float32([markers[0].pos[0], markers[0].pos[1], 0, 0]).reshape(-1, 1)
                    kf.statePost = state
                    found_before = True
                else:
                    kf.correct(markers[0].pos.reshape(-1, 1))
                not_found_count = 0
            else:
                not_found_count += 1
                if not_found_count >= 3:
                    found_before = False
                else:
                    kf.statePost = state"""

            img = frame.copy()
            if found_markers:
                for marker in markers:
                    cv2.circle(img, (marker.pos[0], marker.pos[1]), 5, (255, 0, 0), -1)
            cv2.imshow("kalman", img)
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                break
cam.stop()
