import time
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Thread

import cv2


log_level = logging.DEBUG
default_logger = logging.getLogger('VideoStream (default logger)')
# Logger.setLevel() specifies the lowest-severity log message a logger will handle, where debug is the lowest
# built-in severity level and critical is the highest built-in severity. For example, if the severity level is INFO,
# the logger will handle only INFO, WARNING, ERROR, and CRITICAL messages and will ignore DEBUG messages.
default_logger.setLevel(log_level)
# create console handler and set level handler will send on.
handler = logging.StreamHandler()
handler.setLevel(log_level)
formatter = logging.Formatter('[%(levelname)s] [%(name)s] %(message)s')
handler.setFormatter(formatter)
default_logger.addHandler(handler)


class VideoStream:
    """
    Class that continuously gets frames from a cv2 VideoCapture object
    with a dedicated thread.
    """

    def __init__(self,
                 video_feed_name,
                 src,
                 manual_video_fps,
                 queue_size=3,
                 recording_dir=None,
                 reconnect_threshold_sec=20,
                 do_reconnect=True,
                 resize_fn=None,
                 max_cache=10,
                 logger=None):

        if logger is None:
            self.logger = default_logger
        else:
            self.logger = logger

        self.video_feed_name = video_feed_name  # <cam name>
        self.src = src  # <path>
        self.stream = cv2.VideoCapture(self.src)
        self.reconnect_threshold_sec = reconnect_threshold_sec
        self.do_reconnect = do_reconnect
        self.pauseTime = None
        self.stopped = True
        self.Q = deque(maxlen=queue_size)  # Maximum size of a deque or None if unbounded.
        self.max_cache = max_cache
        self.resize_fn = resize_fn
        self.inited = False
        if (manual_video_fps == -1):
            self.manual_video_fps = None
        else:
            self.manual_video_fps = manual_video_fps
        self.vidInfo = {}

        if recording_dir is not None:
            self.record_source_video = True
            self.recording_dir = Path(recording_dir)
            self.recording_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.record_source_video = False
            self.recording_dir = None
        self.currentFrame = None

    def _init_src(self):
        try:
            self.stream = cv2.VideoCapture(self.src)
            if not self.manual_video_fps:
                self.fps = int(self.stream.get(cv2.CAP_PROP_FPS))
                if self.fps == 0:
                    self.logger.warning('cv2.CAP_PROP_FPS was 0. Defaulting to 30 fps.')
                    self.fps = 30
            else:
                self.fps = self.manual_video_fps
            # width and height returns 0 if stream not captured
            self.vid_width = int(self.stream.get(3))
            self.vid_height = int(self.stream.get(4))

            self.vidInfo = {'video_feed_name': self.video_feed_name, 'height': self.vid_height, 'width': self.vid_width,
                            'manual_fps_inputted': self.manual_video_fps is not None,
                            'fps': self.fps, 'inited': False}

            self.out_vid = None

            if self.vid_width != 0:
                self.inited = True
                self.vidInfo['inited'] = True

            self.__init_src_recorder()

        except Exception as error:
            self.logger.error(f'init stream {self.video_feed_name} error: {error}')

    def __init_src_recorder(self):
        if self.record_source_video and self.inited:
            now = datetime.now()
            day = now.strftime('%Y_%m_%d_%H-%M-%S')
            out_vid_fp = str(self.recording_dir / f'orig_{self.video_feed_name}_{day}.avi')
            print(f'out fp: {out_vid_fp}')
            self.out_vid = cv2.VideoWriter(out_vid_fp, cv2.VideoWriter_fourcc(*'MJPG'), int(
                self.fps), (self.vid_width, self.vid_height))

    def start(self, start_msec=0):
        if not self.inited:
            self._init_src()

        while not self.stream.isOpened():
            self.logger.debug(f'{str(datetime.now())} Connecting to {self.video_feed_name}')
            self.stream = cv2.VideoCapture(self.src)

        if start_msec != 0:
            self.stream.set(cv2.CAP_PROP_POS_MSEC, start_msec)

        self.stopped = False

        t = Thread(target=self._get, args=(), daemon=True)
        t.start()

        self.logger.info(f'Start video streaming for {self.video_feed_name}')
        return self

    def _reconnect_start(self):
        s = Thread(target=self._reconnect, args=(), daemon=True)
        s.start()
        return self

    def _get(self):
        while not self.stopped:
            try:
                if len(self.Q) > self.max_cache:
                    time.sleep(0.01)
                    continue

                grabbed, frame = self.stream.read()
                frame_time = self.get_frame_time()

                if grabbed:
                    self.Q.appendleft((frame, frame_time))

                    if self.record_source_video and self.out_vid:
                        self.out_vid.write(frame)

                    time.sleep(1 / self.fps)

            except Exception as e:
                self.logger.warning(f'Stream {self.video_feed_name} grab error: {e}')
                grabbed = False

            if not grabbed:
                if self.pauseTime is None:
                    self.pauseTime = time.time()
                    self.printTime = time.time()
                    self.logger.info(f'No frames for {self.video_feed_name}, starting {self.reconnect_threshold_sec:0.1f}sec countdown.')
                time_since_pause = time.time() - self.pauseTime
                countdown_time = self.reconnect_threshold_sec - time_since_pause
                time_since_print = time.time() - self.printTime
                if time_since_print > 1 and countdown_time >= 0:  # prints only every 1 sec
                    self.logger.debug(f'No frames for {self.video_feed_name}, countdown: {countdown_time:0.1f}sec')
                    self.printTime = time.time()

                if countdown_time <= 0:
                    if self.do_reconnect:
                        self._reconnect_start()
                        break
                    elif not self._more():
                        self.logger.info('Not reconnecting. Stopping..')
                        self.stop()
                        break
                    else:
                        time.sleep(1)
                        self.logger.debug(f'Countdown reached but still have unconsumed frames in deque: {len(self.Q)}')
                        self.Q.clear()
                continue

            self.pauseTime = None

    def read(self):
        if self._more():
            self.currentFrame, frame_time = self.Q.pop()
        else:
            return [], 0
        if self.resize_fn:
            self.currentFrame = self.resize_fn(self.currentFrame)
        return self.currentFrame, frame_time

    def _more(self):
        return bool(self.Q)

    def stop(self):
        if not self.stopped:
            self.stopped = True
            time.sleep(0.1)

            if self.stream:
                self.stream.release()

            if self._more():
                self.Q.clear()

            if self.out_vid:
                self.out_vid.release()

            self.logger.info(f'Stopped video streaming for {self.video_feed_name}')

    def _reconnect(self):
        self.logger.info(f'Reconnecting to {self.video_feed_name}...')
        if self.stream:
            self.stream.release()

        if self._more():
            self.Q.clear()

        while not self.stream.isOpened():
            self.logger.debug(f'{str(datetime.now())} Reconnecting to {self.video_feed_name}')
            self.stream = cv2.VideoCapture(self.src)
        if not self.stream.isOpened():
            return (f'error opening {self.video_feed_name}')

        if not self.inited:
            self._init_src()

        self.logger.info(f'VideoStream for {self.video_feed_name} initialised!')
        self.pauseTime = None
        self.start()

        return None

    def get_frame_time(self):
        """Get video time of frame.
        
        Returns:
            int: time elapsed since start of video in milliseconds.
        """
        return self.stream.get(cv2.CAP_PROP_POS_MSEC)

    def start_from(self, start_msec):
        self.stop()
        while not self.stream.isOpened():
            self.logger.debug(f'{str(datetime.now())} Connecting to {self.video_feed_name}')
            self.stream = cv2.VideoCapture(self.src)
        self.start(start_msec=start_msec)
