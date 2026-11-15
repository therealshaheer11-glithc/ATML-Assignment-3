"""Small independent controller for bounded GPU sessions and release requests."""
import os
import signal
import threading
import time


class GpuSessionGuard:
    def __init__(self, release_callback, hard_seconds=1200):
        self.release_callback = release_callback
        self.started = time.monotonic()
        self.hard_seconds = hard_seconds
        self.process = None
        self.release_requested = False
        self.release_reason = None
        self.lock = threading.Lock()
        self.timer = threading.Timer(hard_seconds,lambda:self.finish('hard session deadline'))
        self.timer.daemon = True

    def start(self):
        self.timer.start()

    def attach(self, process):
        self.process = process
        if self.release_requested:
            self.stop_worker()
            raise TimeoutError('GPU session expired before its worker started.')

    def stop_worker(self):
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass

    def finish(self, reason='session exit'):
        self.timer.cancel()
        with self.lock:
            if self.release_requested:
                return
            self.release_requested = True
            self.release_reason = reason
        try:
            self.stop_worker()
        finally:
            self.release_callback(reason)
