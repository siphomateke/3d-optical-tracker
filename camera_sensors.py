from urllib2 import urlopen
import json
import numpy as np
import math


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
    def __init__(self, name):
        self.name = name
        self.value = None

    @staticmethod
    def get_last_reading(data):
        return data[len(data) - 1][1]

    def update(self, data):
        self.value = np.array(self.get_last_reading(data))

    def set(self, value):
        self.value = value


def get_jsonparsed_data(url):
    response = urlopen(url)
    data = str(response.read())
    return json.loads(data)


class CameraSensors:
    def __init__(self, url):
        self.url = url # type: str
        self.sensors = {} # type: dict[str, Sensor]

    def update_sensor(self, name, data):
        """ Updates a sensors value. If it doesn't exist, create it"""
        if name not in self.sensors:
            self.sensors[name] = Sensor(name)
        self.sensors[name].update(data[name]["data"])

    def set_sensor(self, name, value):
        if name not in self.sensors:
            self.sensors[name] = Sensor(name)
        self.sensors[name].set(value)

    def get_sensor(self, name):
        """ Return the sensor with the give name """
        if name in self.sensors:
            return self.sensors[name]
        else:
            return False

    def update(self):
        sensors_data = get_jsonparsed_data(self.url + "sensors.json")

        if "accel" in sensors_data:
            self.update_sensor("accel", sensors_data)
            ax, ay, az = self.get_sensor("accel").value
            yaw = math.atan2(ax, ay) * (180 / math.pi)
            pitch = math.atan2(ay, math.sqrt(math.pow(ax, 2) + math.pow(az, 2))) * (180 / math.pi)
            roll = math.atan2(ax, az) * (180 / math.pi)
            self.set_sensor("rotation", yaw)
            self.set_sensor("pitch", pitch)
            self.set_sensor("roll", roll)
