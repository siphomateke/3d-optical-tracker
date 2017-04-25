import serial
import time
from threading import Thread, ThreadError


class SerialStream:
    def __init__(self, ser, connection_char, encoding="utf-8", terminator="\n"):
        self.ser = ser
        self.encoding = encoding
        self.char = connection_char.encode(self.encoding)
        self.thread_cancelled = False
        self.thread = Thread(target=self.run)
        self.data = []
        self.str = ""
        self.terminator = terminator.encode(self.encoding)

    def open(self):
        if not self.ser.is_open:
            self.ser.open()
        self.thread.start()

    def write(self, str):
        self.ser.write(str.encode())

    def flush(self):
        self.ser.flush()

    def run(self):
        while not self.thread_cancelled:
            try:
                self.ser.write(self.char)
                self.ser.flush()
                return_val = self.ser.read(self.ser.in_waiting)
                newline_idx = return_val.find(self.terminator)
                if newline_idx != -1:
                    self.str += return_val[:newline_idx]
                    self.data.append(self.str.decode(self.encoding))
                    self.str = return_val[newline_idx+1:]
                elif len(return_val) > 0:
                    self.str += return_val
            except ThreadError:
                self.thread_cancelled = True

    def get_data(self):
        tmp = self.data
        self.data = []
        return tmp

    def is_running(self):
        return self.thread.isAlive()

    def close(self):
        self.thread_cancelled = True
        # block while waiting for thread to terminate
        while self.thread.isAlive():
            time.sleep(1)
        self.ser.flush()
        if self.ser.is_open:
            self.ser.close()
        return True