import numpy as np
import cv2
import ipcamutil
import math
from urllib2 import urlopen
import json
import serial_utils
import serial


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


class AngleFilter:
    def __init__(self, freq):
        self.freq = freq
        self.last_sine = 0
        self.last_cos = 0
        self.angle = 0

    def update(self, new_angle):
        self.last_sine = self.freq * self.last_sine + (1 - self.freq) * math.sin(new_angle)
        self.last_cos = self.freq * self.last_cos + (1 - self.freq) * math.cos(new_angle)
        self.angle = math.atan2(self.last_sine, self.last_cos)
        return self.angle

    def get_angle(self, degrees=True):
        if degrees:
            return self.angle * (180.0 / math.pi)
        else:
            return self.angle


class Sensor:
    def __init__(self, url):
        self.url = url
        self.accel = None
        self.cur_accel = None
        self.yaw = AngleFilter(0.5)
        self.pitch = AngleFilter(0.5)
        self.roll = AngleFilter(0.5)
        self.translate = 0
        self.G = 9.81
        self.gps = None
        self.network_loc = None
        self.loc = None
        self.mag = None

    def get_sensor_reading(self, sensor):
        return sensor[len(sensor) - 1][1]

    def update(self):
        sensors_data = get_jsonparsed_data(self.url + "sensors.json")
        gps_data = get_jsonparsed_data(self.url + "gps.json")
        if "gps" in gps_data:
            self.gps = gps_data["gps"]
            self.network_loc = gps_data["network"]
            if self.gps["accuracy"] > self.network_loc["accuracy"]:
                self.loc = [self.gps["latitude"], self.gps["longitude"], self.gps["altitude"]]
            else:
                self.loc = [self.network_loc["latitude"], self.network_loc["longitude"], self.gps["altitude"]]
        self.mag = sensors_data["mag"]["data"]
        self.cur_mag = self.get_sensor_reading(self.mag)
        self.accel = sensors_data["accel"]["data"]
        self.cur_accel = self.accel[len(self.accel) - 1][1]
        ax = self.cur_accel[0]
        ay = self.cur_accel[1]
        az = self.cur_accel[2]
        self.G = math.sqrt((math.pow(ax, 2) + math.pow(ay, 2) + math.pow(az, 2)))

        new_yaw = math.atan2(ax, ay)
        new_pitch = math.atan2(ay, math.sqrt(math.pow(ax, 2) + math.pow(az, 2)))
        self.pitch2 = (-(math.acos(ay / self.G) - (math.pi / 2.0))) * (180.0 / math.pi)
        new_roll = math.atan2(ax, az)
        self.yaw.update(new_yaw)
        self.pitch.update(new_pitch)
        self.roll.update(new_roll)
        self.translate += ((self.cur_accel[2] / self.G) - self.translate) / 4.0

    def list_to_str(self, list):
        str = ""
        for i in range(len(list)):
            str += "{}".format(list[i])
            if i < len(list) - 1:
                str += ","
        return str

    def display(self):
        img = np.zeros((600, 600, 3), dtype=np.uint8)
        cv2.rectangle(img, (150, 150), (450, 450), (0, 0, 255), 2)
        cv2.circle(img, (int((self.cur_accel[0] / self.G) * -150) + 300, int((self.cur_accel[1] / self.G) * 150) + 300),
                   5, (0, 255, 255), -1)
        cv2.circle(img, (int(self.roll.get_angle() / 180 * -300) + 300, int(self.pitch.get_angle() / 180 * 300) + 300),
                   5, (0, 255, 0), -1)
        cv2.circle(img, (int(self.roll.get_angle() / 180 * -300) + 300, int(self.pitch2 / 180 * 300) + 300),
                   5, (255, 0, 0), -1)
        cv2.circle(img, (int(self.tilt / 360 * 600), 300), 5, (255, 255, 0), -1)
        cv2.imshow("Accel", img)


ipcamera_url = "http://192.168.8.100:8080/"
cam = ipcamutil.Cam(ipcamera_url + "video")
cam.start()
sensor = Sensor(ipcamera_url)
sensor.tilt = 0

then = cv2.getTickCount()
ser = serial_utils.SerialStream(serial.Serial(baudrate=9600, port="COM3", timeout=0), "a")
ser.open()

timer = 0

while True:
    now = cv2.getTickCount()
    delta = ((now - then) / cv2.getTickFrequency())
    then = now

    sensor.update()
    timer += delta
    if timer / 1000.0 > 25:
        ser.write("m" + sensor.list_to_str(sensor.cur_mag))
        ser.flush()
    for line in ser.get_data():
        orientation = line.split(",")
        sensor.tilt = float(orientation[0])
        print sensor.tilt
    if cam.is_opened():
        frame = cam.get_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        transformed = rotate_bound(frame, -sensor.yaw.get_angle() + 90)
        cv2.imshow('Camera', transformed)

        sensor.display()
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

ser.close()
cam.shut_down()
cv2.destroyAllWindows()
