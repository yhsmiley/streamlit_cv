import atexit
import logging
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import pkg_resources
import streamlit as st
import torch

from bytetrack_realtime.byte_tracker import ByteTracker
from deep_sort_realtime.deepsort_tracker import DeepSort
from misc.utils import annotate_image, annotate_tracks, annotate_byte_tracks, process_image, process_tracks, process_byte_tracks, COCO_CLASSES
from misc.video_getter_cv2 import VideoStream
from scaledyolov4.scaled_yolov4 import ScaledYOLOV4


# set up logging (inherit from streamlit's logger)
log_level = logging.INFO
logger = logging.getLogger(__name__)
logger.setLevel(log_level)


def clear_cuda_cache():
    torch.cuda.empty_cache()


@st.cache(allow_output_mutation=True)
def initialize_od(model_architecture, input_size):
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

    return ScaledYOLOV4(
        bgr=True,
        gpu_device=0,
        model_image_size=input_size,
        max_batch_size=1,
        half=True,
        same_size=True,
        weights=weights,
        cfg=cfg,
    )


@st.cache(allow_output_mutation=True)
def initialize_deepsort():
    return DeepSort(
        max_age=30,
        nn_budget=10,
        embedder='clip_ViT-B/32'
    )


@st.cache(allow_output_mutation=True)
def initialize_bytetracker():
    return ByteTracker(
        track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.9
    )


def initialize_vid_getter(stream_source, video_feed_name='video1', manual_video_fps=-1,
                          do_reconnect=False, reconnect_threshold_sec=3, queue_size=3):
    return VideoStream(
        video_feed_name,
        stream_source,
        manual_video_fps=manual_video_fps,
        reconnect_threshold_sec=reconnect_threshold_sec,
        do_reconnect=do_reconnect,
        queue_size=queue_size
    )


def get_vid_time_from_sec(sec):
    vid_duration_sec = timedelta(seconds=sec)
    return datetime.strptime(str(vid_duration_sec), "%H:%M:%S")


def initialize_video(stream_source):
    st.session_state.vid_getter = initialize_vid_getter(stream_source)
    vid_frame_count = int(st.session_state.vid_getter.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = int(st.session_state.vid_getter.stream.get(cv2.CAP_PROP_FPS))
    st.session_state.inited = True
    return get_vid_time_from_sec(int(vid_frame_count / vid_fps))


def get_msec_from_vid_start_time():
    # 16hr due to streamlit timezone bug
    return timedelta(
        hours=st.session_state.vid_start_time.hour - 16,
        minutes=st.session_state.vid_start_time.minute,
        seconds=st.session_state.vid_start_time.second
    ).total_seconds() * 1000


def hex_to_bgr(hex_string):
    r_hex = hex_string[1:3]
    g_hex = hex_string[3:5]
    b_hex = hex_string[5:7]
    return int(b_hex, 16), int(g_hex, 16), int(r_hex, 16)


def cleanup(tfile_path):
    logger.info('Cleaning up...')
    Path(tfile_path).unlink()


def main():
    def frame_chooser_on_change(tracker_deepsort, tracker_bytetrack):
        if not st.session_state.stopped:
            st.session_state.vid_getter.start_from(start_msec=get_msec_from_vid_start_time())
            tracker_deepsort.delete_all_tracks()
            tracker_bytetrack.delete_all_tracks()

    def tracking_on_change(tracker_deepsort, tracker_bytetrack):
        clear_cuda_cache()
        if not st.session_state.stopped:
            tracker_deepsort.delete_all_tracks()
            tracker_bytetrack.delete_all_tracks()


    st.set_page_config(layout='wide')
    st.title('Object detection with ScaledYOLOv4 (Video File)')

    confidence_threshold = st.sidebar.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.05)
    nms_threshold = st.sidebar.slider('NMS threshold', 0.0, 1.0, 0.5, 0.05)

    model_architecture = st.sidebar.selectbox('Scaled-YOLOv4 model', ('csp', 'p5', 'p6', 'p7'), index=1, on_change=clear_cuda_cache)
    input_size = st.sidebar.selectbox('Model input size', (512, 640, 896, 1280, 1536), index=2, on_change=clear_cuda_cache)
    classes = st.sidebar.multiselect('Classes to display', COCO_CLASSES, ['person'])

    od = initialize_od(model_architecture, input_size)
    tracker_deepsort = initialize_deepsort()
    tracker_bytetrack = initialize_bytetracker()

    tracking = st.sidebar.radio("Tracking?", ('None', 'DeepSORT', 'ByteTrack'), on_change=tracking_on_change, args=(tracker_deepsort, tracker_bytetrack))
    color_hex = st.sidebar.color_picker('Color for bbox', '#0000ff')
    bbox_color = hex_to_bgr(color_hex)
    font_size = st.sidebar.slider('Font size', 0.0, 1.5, 1.0, 0.1)

    if 'stream_source' not in st.session_state:
        st.session_state.stream_source = None

    stream_upload = st.empty()
    vid_file_buffer = None
    vid_file_buffer = stream_upload.file_uploader('Upload a video file', type=['mp4', 'avi', 'mpeg', 'mov', 'mkv'])
    if vid_file_buffer is not None:
        if st.session_state.stream_source is None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(vid_file_buffer.read())
            st.session_state.stream_source = tfile.name
            atexit.register(cleanup, tfile.name)
    else:
        st.session_state.stream_source = None
        st.session_state.inited = False

    initialize_vid = st.empty()
    img_placeholder = st.empty()
    frame_time = st.empty()
    frame_chooser = st.empty()
    start_stop = st.empty()

    if 'inited' not in st.session_state:
        st.session_state.inited = False

    if st.session_state.stream_source is not None and not st.session_state.inited and initialize_vid.button('Initialize Video'):
        st.session_state.vid_duration = initialize_video(st.session_state.stream_source)
    if st.session_state.inited:
        tz_offset = timedelta(hours=8)
        frame_chooser.slider(
            'Starting video time',
            min_value=(datetime.strptime("0:0:0", "%H:%M:%S") - tz_offset).time(),
            max_value=(st.session_state.vid_duration - tz_offset).time(),
            step=timedelta(seconds=1),
            format='HH:mm:ss',
            key='vid_start_time',
            on_change=frame_chooser_on_change,
            args=(tracker_deepsort, tracker_bytetrack)
        )

    if 'stopped' not in st.session_state:
        st.session_state.stopped = True

    if st.session_state.inited and start_stop.button('Start / Stop'):
        if st.session_state.stopped:
            if st.session_state.stream_source is None:
                st.error('Set a streaming source!')
                st.stop()
            else:
                st.session_state.stopped = False
                st.session_state.vid_getter.start(start_msec=get_msec_from_vid_start_time())
                time.sleep(0.5)
        else:
            st.session_state.stopped = True
            st.session_state.vid_getter.stop()
            time.sleep(0.5)

    while not st.session_state.stopped:
        try:
            vid_frame, curr_vid_time_msec = st.session_state.vid_getter.read()

            if len(vid_frame):
                if tracking == 'DeepSORT':
                    tracks = process_tracks(od, tracker_deepsort, vid_frame, confidence_threshold, nms_threshold, classes=classes)
                    img = annotate_tracks(vid_frame, tracks, color=bbox_color, font_size=font_size)
                elif tracking == 'ByteTrack':
                    tracks = process_byte_tracks(od, tracker_bytetrack, vid_frame, nms_threshold, low_confidence_threshold=0.1, classes=classes)
                    img = annotate_byte_tracks(vid_frame, tracks, color=bbox_color, font_size=font_size)
                else:
                    detections = process_image(od, vid_frame, confidence_threshold, nms_threshold, classes=classes)
                    img = annotate_image(vid_frame, detections, color=bbox_color, font_size=font_size)
                img_placeholder.image(img, channels='BGR')
                frame_time.write(f'Current video time: {get_vid_time_from_sec(int(curr_vid_time_msec/1000)).time()}')

            elif st.session_state.vid_getter.stopped:
                st.error('Video file ended. Stopping...')
                st.session_state.stopped = True
                time.sleep(0.5)
                break

        except KeyboardInterrupt:
            st.session_state.vid_getter.stop()
            time.sleep(0.5)
            st.session_state.stopped = True


if __name__ == '__main__':
    main()
