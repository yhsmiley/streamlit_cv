import cv2
import numpy as np
import pkg_resources
import streamlit as st
import torch

from misc.utils import process_image, annotate_image, COCO_CLASSES
from scaledyolov4.scaled_yolov4 import ScaledYOLOV4


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


def hex_to_bgr(hex_string):
    r_hex = hex_string[1:3]
    g_hex = hex_string[3:5]
    b_hex = hex_string[5:7]
    return int(b_hex, 16), int(g_hex, 16), int(r_hex, 16)


def main():
    st.set_page_config(layout='wide')
    st.title('Object detection with ScaledYOLOv4 (Single image)')

    confidence_threshold = st.sidebar.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.05)
    nms_threshold = st.sidebar.slider('NMS threshold', 0.0, 1.0, 0.5, 0.05)

    model_architecture = st.sidebar.selectbox('Scaled-YOLOv4 model', ('csp', 'p5', 'p6', 'p7'), index=1, on_change=clear_cuda_cache)
    input_size = st.sidebar.selectbox('Model input size', (512, 640, 896, 1280, 1536), index=2, on_change=clear_cuda_cache)
    classes = st.sidebar.multiselect('Classes to display', COCO_CLASSES, ['person'])

    color_hex = st.sidebar.color_picker('Color for bbox', '#0000ff')
    bbox_color = hex_to_bgr(color_hex)
    font_size = st.sidebar.slider('Font size', 0.5, 1.5, 1.0, 0.1)

    od = initialize_od(model_architecture, input_size)

    img_file_buffer = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

    if img_file_buffer is not None:
        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        detections = process_image(od, image, confidence_threshold, nms_threshold, classes=classes)
        image = annotate_image(image, detections, color=bbox_color, font_size=font_size)

        st.image(image, caption='Processed image', use_column_width=True, channels='BGR')


if __name__ == '__main__':
    main()
