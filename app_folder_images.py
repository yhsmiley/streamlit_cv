from pathlib import Path

import cv2
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
        max_batch_size=16,
        half=True,
        same_size=True,
        weights=weights,
        cfg=cfg,
    )



def main():
    st.set_page_config(layout='wide')
    st.title('Object detection with ScaledYOLOv4 (Folder of images)')
    images_folder = st.text_input('Folder path containing images')
    confidence_threshold = st.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.05)
    nms_threshold = st.slider('NMS threshold', 0.0, 1.0, 0.5, 0.05)
    num_cols = st.slider('Number of display columns', 1, 5, 2, 1)

    model_architecture = st.selectbox('Scaled-YOLOv4 model', ('csp', 'p5', 'p6', 'p7'), index=1)
    input_size = st.selectbox('Model input size', (512, 640, 896, 1280, 1536), index=2)
    od = initialize_od(model_architecture, input_size)

    images_folder = str(images_folder) or None

    suffixes = ['.png', '.jpg', '.jpeg']
    if images_folder is not None:
        if not Path(images_folder).is_dir():
            st.error('Folder does not exist!')
        else:
            all_images = []
            for image_path in Path(images_folder).glob('*'):
                if image_path.suffix in suffixes:
                    image = cv2.imread(str(image_path))

                    detections = process_image_cache(od, image, confidence_threshold, nms_threshold)
                    img = annotate_image_cache(image, detections)

                    all_images.append(img)

            for i in range(0, len(all_images), num_cols):
                cols = st.columns(num_cols)
                images = all_images[i:i+num_cols]
                for j, _ in enumerate(images):
                    cols[j].image(images[j], channels='BGR')


if __name__ == '__main__':
    main()
