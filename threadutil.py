import time
from threading import Thread, ThreadError


class ProgramThread:
    def __init__(self, func):
        self.func = func
        self.thread_cancelled = False
        self.thread = Thread(target=self._run)

    def start_thread(self):
        self.thread.start()

    def _run(self):
        while not self.thread_cancelled:
            try:
                self.func()
            except ThreadError:
                self.thread_cancelled = True

    def is_running(self):
        return self.thread.isAlive()

    def stop_thread(self):
        self.thread_cancelled = True
        # block while waiting for thread to terminate
        """while self.is_running():
            time.sleep(1)"""
        self.thread.join()
        return True
