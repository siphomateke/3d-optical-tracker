import numpy as np
import cv2
import ipcamutil

from visual_odometry import PinholeCamera, VisualOdometry
from urllib2 import urlopen
import json


def get_jsonparsed_data(url):
    response = urlopen(url)
    data = str(response.read())
    return json.loads(data)


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    """# compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY"""

    nW = image.shape[1]
    nH = image.shape[0]

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


class Sensor:
    def __init__(self, url):
        self.url = url
        self.accel = None
        self.cur_accel = None
        self.rotation = 0
        self.translate = 0
        self.G = 9.81
        self.gps = None
        self.network_loc = None
        self.loc = None

    def update(self):
        sensors_data = get_jsonparsed_data(self.url + "sensors.json")
        gps_data = get_jsonparsed_data(self.url + "gps.json")
        self.gps = gps_data["gps"]
        self.network_loc = gps_data["network"]
        if self.gps["accuracy"] > self.network_loc["accuracy"]:
            self.loc = [self.gps["latitude"], self.gps["longitude"], self.gps["altitude"]]
        else:
            self.loc = [self.network_loc["latitude"], self.network_loc["longitude"], self.gps["altitude"]]
        print self.loc
        self.accel = sensors_data["accel"]["data"]
        self.cur_accel = self.accel[len(self.accel) - 1][1]
        # Dampen
        self.rotation += ((self.cur_accel[0] / self.G) - self.rotation) / 4.0
        self.translate += ((self.cur_accel[2] / self.G) - self.translate) / 4.0

    def display(self):
        img = np.zeros((600, 600, 3), dtype=np.uint8)
        cv2.circle(img, (int((self.cur_accel[0] / self.G) * -300) + 300, int((self.cur_accel[1] / self.G) * 300) + 300), 5, (0, 0, 255), -1)
        cv2.circle(img, (int((self.cur_accel[0] / self.G) * -300) + 300, int((self.cur_accel[2] / self.G) * 300) + 300),
                   5, (0, 255, 0), -1)
        cv2.imshow("Accel", img)


camera_name = "LG-K8"
cam = PinholeCamera("camera/" + camera_name + ".npz")
vo = VisualOdometry(cam)

traj = np.zeros((600, 600, 3), dtype=np.uint8)

ipcamera_url = "http://192.168.8.100:8080/"
cam = ipcamutil.Cam(ipcamera_url + "video")
cam.start()
sensor = Sensor(ipcamera_url)

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

        # vo.update(gray, delta)

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
        transformed = rotate_bound(frame, (sensor.rotation * -90) + 90)
        cv2.imshow('Camera', transformed)
        cv2.imshow('Trajectory', traj)

        sensor.display()
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
cam.shut_down()
cv2.destroyAllWindows()

cv2.imwrite('map.png', traj)
