import threading
import cv2
import pyrealsense2 as rs
import numpy as np
from queue import Queue, Empty
from utilities.hough_transform_lane_detection import HoughTransformLaneDetection


# Functions to read frames from Video or RealSense Camera
class VideoCaptureLatestFrame:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.q = Queue(maxsize=0)
        t = threading.Thread(target=self._reader)
        # when worker threads are daemon threads, they die when all non-daemon threads (e.g. main thread) have exited.
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one in queue
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print('stopping the thread as there is no frame returned')
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except Empty:
                    pass
            self.q.put(frame)

    def read(self):
        try:
            return self.q.get(timeout=10)
        except Empty:
            print("No value in queue for 10 sec")
            return None

    def get_cap(self):
        return self.cap

    def release(self):
        self.cap.release()



class VideoCaptureLatestFrameFromRealSense:
    def __init__(self, width=1280, height=720, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)

        self.q = Queue(maxsize=0)
        t = threading.Thread(target=self._reader)
        # when worker threads are daemon threads, they die when all non-daemon threads (e.g. main thread) have exited.
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one in queue
    def _reader(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except Empty:
                    pass
            self.q.put(color_image)

    def read(self):
        try:
            return self.q.get(timeout=10)
        except Empty:
            print("No value in queue for 10 sec")
            return None

    def get_cap(self):
        return self.pipeline

    def release(self):
        self.pipeline.stop()


class VideoCaptureNextFrame:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)

    def read(self):
        ret, frame = self.cap.read()
        # ret is false when camera is disconnected OR no more frames in video file
        if ret:
            return frame
        return None

    def get_cap(self):
        return self.cap

    def release(self):
        self.cap.release()


def runPipelineOnVideo(videofile):
    try:
        vid_capture = None
        if videofile is None:
            print("with Camera")
            vid_capture = VideoCaptureLatestFrame()
        else:
            print("With Input Video File: {}".format(videofile))
            vid_capture = VideoCaptureNextFrame(videofile)

        cap = vid_capture.get_cap()
        input_video_frames_per_second = int(cap.get(cv2.CAP_PROP_FPS))
        output_video_frames_per_second = input_video_frames_per_second
        print("Input TotalFrames: {} | WidthXHeight: {}X{} | FPS:{}".format(
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            input_video_frames_per_second))

        pipeline = HoughTransformLaneDetection(False, False,
                                               False, True,
                                               output_video_frames_per_second, int(cap.get(3)), int(cap.get(4)))

        while True:
            frame = vid_capture.read()
            if frame is None:
                print("No frame received. Closing.")
                break
            error, delta, output = pipeline.processFrame(frame)
            # cv2.imshow('output', output)
            #
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    finally:
        pipeline.closePipeline()
        vid_capture.get_cap().release()
        cv2.destroyAllWindows()
