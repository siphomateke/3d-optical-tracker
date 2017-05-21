import cv2
import numpy as np

MIN_AREA = 1
NUM_PXS = 200
MIN_THRESH = 160
MULTIPLE_MARKERS = False


# TODO: Add multiple marker tracking


class Marker:
    def __init__(self, pos, id=-1):
        self.pos = pos
        self.new_pos = pos
        self.id = id
        self.detected = False
        self.found = False
        self.not_found_count = 0

        """self.dynamParams = 4
        self.measureParams = 2
        self.kf = cv2.KalmanFilter(self.dynamParams, self.measureParams)

        self.kf.transitionMatrix = np.eye(self.dynamParams, dtype=np.float32)  # A

        self.kf.measurementMatrix = np.zeros((self.measureParams, self.dynamParams), dtype=np.float32)  # H
        self.kf.measurementMatrix[0, 0] = 1
        self.kf.measurementMatrix[1, 1] = 1

        self.kf.processNoiseCov = np.eye(self.dynamParams, dtype=np.float32) * 1e-2  # Q
        self.kf.processNoiseCov[2, 2] = 1.0
        self.kf.processNoiseCov[3, 3] = 1.0

        self.kf.measurementNoiseCov = np.eye(self.measureParams, dtype=np.float32) * 1e-3  # Q
        self.kf.errorCovPre = np.eye(self.dynamParams, dtype=np.float32) * 0  # P

        # Initial state
        self.kf.statePost = np.float32([self.pos[0], self.pos[1], 0, 0]).reshape(-1, 1)"""

    @property
    def x(self):
        return self.pos.ravel()[0]

    @property
    def y(self):
        return self.pos.ravel()[1]

    def predict(self):
        pass

    def set_marker(self, marker):
        self.detected = True
        self.new_pos = marker.pos

    def update(self, time):
        if self.found:
            self.kf.transitionMatrix[0, 2] = time
            self.kf.transitionMatrix[1, 3] = time
            state = self.kf.predict()
            self.pos = state[:2]

        # TODO: rename detected, found and not_found_count
        if self.detected:
            self.not_found_count = 0
            if not self.found:
                state = np.float32([self.new_pos[0], self.new_pos[1], 0, 0]).reshape(-1, 1)
                self.kf.statePost = state
                self.found = True
            else:
                self.kf.correct(self.new_pos.reshape(-1, 1))
        else:
            self.not_found_count += 1
            if self.not_found_count >= 3:
                self.found = False
            """else:
                # TODO: check if this is needed
                self.kf.statePost = state"""

        self.detected = False

    def dist_to(self, marker):
        return np.linalg.norm(self.pos - marker.pos)


class MarkerTracker:
    def __init__(self):
        self.markers = {}
        self.last_id = 0
        self.MAX_MARKER_DIST = 100
        self.then = -1
        self.now = -1

    def update(self, markers):
        """
        Updates the marker tracker creating new markers if necessary and updating old ones
        :param markers: The current markers
        :type markers: list[Marker]
        """
        self.now = cv2.getTickCount()
        if self.then == -1:
            self.then = self.now
        time = (self.now - self.then) / cv2.getTickFrequency()  # seconds
        self.then = self.now

        for new_marker in markers:
            best_id = -1
            min_dist = -1
            # Find closest marker
            for id in self.markers:
                old_marker = self.markers[id]
                dist = new_marker.dist_to(old_marker)
                if dist < min_dist or min_dist == -1:
                    min_dist = dist
                    best_id = id
            # This marker is already tracked
            # if min_dist != -1 and min_dist < self.MAX_MARKER_DIST:
            if min_dist != -1:
                self.markers[best_id].set_marker(new_marker)
            # Otherwise create new marker
            else:
                self.markers[self.last_id] = new_marker
                self.markers[self.last_id].id = self.last_id
                self.last_id += 1

        for id in self.markers:
            self.markers[id].update(time)

    def get_markers(self):
        return [self.markers[key] for key in self.markers]


class MarkerFinder:
    def __init__(self):
        self.marker_tracker = MarkerTracker()

    def find_markers(self, image):
        """
        Finds bright spots in an image which are likely to be markers
        :param image:
        :return:
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
        # gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # cv2.imshow("Gray", gray)

        # Find brightest pixels
        _, thresh_mask = cv2.threshold(gray, MIN_THRESH, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel)

        cv2.imshow("Thresh", thresh_mask)

        max_search = cv2.bitwise_and(gray, gray, mask=thresh_mask)
        # cv2.imshow("Max search", max_search)

        thresh = np.zeros(gray.shape, np.uint8)
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(max_search)
        mask = (max_search > max_val * 0.8) & (max_search > MIN_THRESH)
        thresh[mask] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("Max thresh", thresh)

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.array(contours)
        markers = []
        if len(contours) > 0:
            areas = np.array([cv2.contourArea(c) for c in contours])
            contours = contours[np.argsort(areas)[::-1]]

            def add_marker(c):
                area = cv2.contourArea(c)
                if area > MIN_AREA:
                    M = cv2.moments(c)
                    x = float(M['m10']) / M['m00']
                    y = float(M['m01']) / M['m00']
                    markers.append(Marker(np.float32([x, y])))

            if MULTIPLE_MARKERS:
                for c in contours:
                    add_marker(c)
            else:
                c = contours[0]
                add_marker(c)

        # self.marker_tracker.update(markers)
        # marker_tracker_markers = self.marker_tracker.get_markers()

        # return len(marker_tracker_markers) > 0, marker_tracker_markers
        return len(markers) > 0, markers
