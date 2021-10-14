import cv2
import numpy as np
import pkg_resources
import streamlit as st

from misc.utils import process_image_cache, annotate_image_cache
from scaledyolov4.scaled_yolov4 import ScaledYOLOV4


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


def main():
    st.set_page_config(layout='wide')
    st.title('Object detection with ScaledYOLOv4')
    img_file_buffer = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
    confidence_threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.05)
    nms_threshold = st.slider('NMS threshold', 0.0, 1.0, 0.5, 0.05)

    model_architecture = st.selectbox('Scaled-YOLOv4 model', ('csp', 'p5', 'p6', 'p7'), index=1)
    input_size = st.selectbox('Model input size', (512, 640, 896, 1280, 1536), index=2)
    od = initialize_od(model_architecture, input_size)

    if img_file_buffer is not None:
        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        detections = process_image_cache(od, image, confidence_threshold, nms_threshold)
        # image, labels = annotate_image(image, detections)
        image = annotate_image_cache(image, detections)

        st.image(image, caption='Processed image', use_column_width=True, channels='BGR')
        # st.write(labels)


if __name__ == '__main__':
    main()
