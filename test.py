import numpy as np
import cv2
import ipcamutil
from camera_sensors import CameraSensors
from imutils import rotate_img

from visual_odometry import PinholeCamera, VisualOdometry
from urllib2 import urlopen
import json


camera_name = "LG-K8"
cam = PinholeCamera("camera/" + camera_name + ".npz")
vo = VisualOdometry(cam)

traj = np.zeros((600, 600, 3), dtype=np.uint8)

ipcamera_url = "http://192.168.8.100:8080/"
cam = ipcamutil.Cam(ipcamera_url)
cam.start()
sensor = CameraSensors(ipcamera_url)

prev_draw_x, prev_draw_y = 300, 300
then = cv2.getTickCount()
while True:
    sensor.update()
    if cam.is_opened():
        frame = cam.get_frame()
        if vo.optical_flow is None:
            vo.optical_flow = np.zeros_like(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        now = cv2.getTickCount()
        delta = ((now - then) / cv2.getTickFrequency())
        then = now

        vo.update(gray, delta)

        cur_t = vo.cur_t
        if vo.frame_stage > 1:
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
        else:
            x, y, z = 0., 0., 0.
        draw_scale = 10
        draw_x, draw_y = int(x * draw_scale) + 300, int(z * draw_scale) + 300

        # cv2.circle(traj, (draw_x, draw_y), 1, (255, 0, 0), -1)
        cv2.line(traj, (draw_x, draw_y), (prev_draw_x, prev_draw_y), (255, 0, 0), 1)
        prev_draw_x = draw_x
        prev_draw_y = draw_y
        cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
        cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        if vo.optical_flow is not None:
            lines = cv2.add(frame, vo.optical_flow)
            cv2.imshow("Lines", lines)
        cv2.imshow('Trajectory', traj)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
cam.shut_down()
cv2.destroyAllWindows()

cv2.imwrite('map.png', traj)
