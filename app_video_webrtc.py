import logging
from pathlib import Path

import av
import pkg_resources
import streamlit as st
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from typing import Tuple

from misc.utils import process_image, annotate_image
from scaledyolov4.scaled_yolov4 import ScaledYOLOV4


# set up logging (inherit from streamlit's logger)
log_level = logging.INFO
logger = logging.getLogger(__name__)
logger.setLevel(log_level)


DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_NMS_THRESHOLD = 0.5
DEFAULT_MODEL_ARCHITECTURE = 'p5'
DEFAULT_INPUT_SIZE = 896
# RTC_CONFIGURATION = RTCConfiguration()
RTC_CONFIGURATION = RTCConfiguration({'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]})


# class ScaledYOLOv4VideoProcessor(VideoProcessorBase):
#     confidence_threshold: float
#     nms_threshold: float

#     def __init__(self) -> None:
#         self.od = ScaledYOLOV4(
#             bgr=True,
#             gpu_device=0,
#             thresh=DEFAULT_CONFIDENCE_THRESHOLD,
#             nms_thresh=DEFAULT_NMS_THRESHOLD,
#             model_image_size=896,
#             max_batch_size=1,
#             half=True,
#             same_size=True,
#             weights=pkg_resources.resource_filename('scaledyolov4', 'weights/yolov4-p5_-state.pt'),
#             cfg=pkg_resources.resource_filename('scaledyolov4', 'configs/yolov4-p5.yaml'),
#         )
#         self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
#         self.nms_threshold = DEFAULT_NMS_THRESHOLD

#     def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#         image = frame.to_ndarray(format='bgr24')
#         detections = process_image(self.od, image, self.confidence_threshold, self.nms_threshold)
#         annotated_image = annotate_image(image, detections)

#         return av.VideoFrame.from_ndarray(annotated_image, format='bgr24')

class ScaledYOLOv4VideoProcessor(VideoProcessorBase):
    confidence_threshold: float
    nms_threshold: float

    def __init__(self) -> None:
        weights, cfg = self._check_model_architecture(DEFAULT_MODEL_ARCHITECTURE)

        self.od = ScaledYOLOV4(
            bgr=True,
            gpu_device=0,
            thresh=DEFAULT_CONFIDENCE_THRESHOLD,
            nms_thresh=DEFAULT_NMS_THRESHOLD,
            model_image_size=DEFAULT_INPUT_SIZE,
            max_batch_size=1,
            half=True,
            same_size=True,
            weights=weights,
            cfg=cfg,
        )
        self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        self.nms_threshold = DEFAULT_NMS_THRESHOLD

    @staticmethod
    def _check_model_architecture(model_architecture: str) -> Tuple[str, str]:
        if model_architecture == 'csp':
            weights = pkg_resources.resource_filename('scaledyolov4', 'weights/yolov4l-mish_-state.pt')
            cfg = pkg_resources.resource_filename('scaledyolov4', 'configs/yolov4-csp.yaml')
        elif model_architecture == 'p5':
            weights = pkg_resources.resource_filename('scaledyolov4', 'weights/yolov4-p5_-state.pt')
            cfg = pkg_resources.resource_filename('scaledyolov4', 'configs/yolov4-p5.yaml')
        elif model_architecture == 'p6':
            weights = pkg_resources.resource_filename('scaledyolov4', 'weights/yolov4-p6_-state.pt')
            cfg = pkg_resources.resource_filename('scaledyolov4', 'configs/yolov4-p6.yaml')
        elif model_architecture == 'p7':
            weights = pkg_resources.resource_filename('scaledyolov4', 'weights/yolov4-p7-state.pt')
            cfg = pkg_resources.resource_filename('scaledyolov4', 'configs/yolov4-p7.yaml')
        return weights, cfg

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format='bgr24')
        if self.od is not None:
            detections = process_image(self.od, image, self.confidence_threshold, self.nms_threshold)
            annotated_image = annotate_image(image, detections)
        else:
            annotated_image = image

        return av.VideoFrame.from_ndarray(annotated_image, format='bgr24')

    def change_od(self, model_architecture: str, input_size: int) -> None:
        weights, cfg = self._check_model_architecture(model_architecture)
        logger.info(f'changing model to {model_architecture}, size {input_size}')

        del self.od
        self.od = None

        self.od = ScaledYOLOV4(
            bgr=True,
            gpu_device=0,
            thresh=DEFAULT_CONFIDENCE_THRESHOLD,
            nms_thresh=DEFAULT_NMS_THRESHOLD,
            model_image_size=input_size,
            max_batch_size=1,
            half=True,
            same_size=True,
            weights=weights,
            cfg=cfg,
        )


def main():
    def create_player():
        return MediaPlayer(stream_source)

    def in_recorder_factory():
        if recording_dir is not None:
            return MediaRecorder(f'{recording_dir}/input.mp4', format='flv')
        return None

    def out_recorder_factory():
        if recording_dir is not None:
            return MediaRecorder(f'{recording_dir}/output.mp4', format='flv')
        return None

    st.set_page_config(layout='wide')
    st.title('Object detection with ScaledYOLOv4 (Video)')
    recording_dir = st.text_input('Folder to save recorded videos')
    confidence_threshold = st.empty()
    nms_threshold = st.empty()

    if recording_dir is not None:
        Path(recording_dir).mkdir(parents=True, exist_ok=True)

    model_architecture = st.selectbox('Scaled-YOLOv4 model', ('csp', 'p5', 'p6', 'p7'), index=1)
    input_size = st.selectbox('Model input size', (512, 640, 896, 1280, 1536), index=2)

    source_type = st.radio('Streaming source type', ('webcam', 'file', 'rtsp/http'))
    stream_upload = st.empty()
    if source_type == 'file':
        stream_source = stream_upload.file_uploader('Upload a video file', type=['mp4', 'avi', 'mpeg', 'mov', 'mkv'])
        if stream_source is not None:
            webrtc_ctx = webrtc_streamer(
                key=f'file-{stream_source}',
                mode=WebRtcMode.RECVONLY,
                # rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={'video': True, 'audio': False},
                player_factory=create_player,
                video_processor_factory=ScaledYOLOv4VideoProcessor,
                in_recorder_factory=in_recorder_factory,
                out_recorder_factory=out_recorder_factory,
            )
        else:
            st.stop()

    elif source_type == 'rtsp/http':
        stream_source = str(stream_upload.text_input('rtsp/http'))
        if not stream_source.startswith(('rtsp', 'http')):
            st.error('Stream name should start with rtsp or http!')
            st.stop()

        webrtc_ctx = webrtc_streamer(
            key=f'stream-{stream_source}',
            mode=WebRtcMode.RECVONLY,
            # rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={'video': True, 'audio': False},
            player_factory=create_player,
            video_processor_factory=ScaledYOLOv4VideoProcessor,
            in_recorder_factory=in_recorder_factory,
            out_recorder_factory=out_recorder_factory,
        )

    elif source_type == 'webcam':
        webrtc_ctx = webrtc_streamer(
            key='object-detection',
            mode=WebRtcMode.SENDRECV,
            # rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={'video': True, 'audio': False},
            video_processor_factory=ScaledYOLOv4VideoProcessor,
            in_recorder_factory=in_recorder_factory,
            out_recorder_factory=out_recorder_factory,
        )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold.slider('Confidence threshold', 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)
        webrtc_ctx.video_processor.nms_threshold = nms_threshold.slider('NMS threshold', 0.0, 1.0, DEFAULT_NMS_THRESHOLD, 0.05)

        if model_architecture or input_size:
            webrtc_ctx.video_processor.change_od(model_architecture, input_size)


if __name__ == '__main__':
    main()
