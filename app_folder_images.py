from pathlib import Path

import cv2
import pkg_resources
import streamlit as st

from misc.utils import process_image_cache, annotate_image_cache, COCO_CLASSES
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


def hex_to_bgr(hex_string):
    r_hex = hex_string[1:3]
    g_hex = hex_string[3:5]
    b_hex = hex_string[5:7]
    return int(b_hex, 16), int(g_hex, 16), int(r_hex, 16)


def main():
    st.set_page_config(layout='wide')
    st.title('Object detection with ScaledYOLOv4 (Folder of images)')

    confidence_threshold_col, _, nms_threshold_col = st.columns([5, 1, 5])
    confidence_threshold = confidence_threshold_col.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.05)
    nms_threshold = nms_threshold_col.slider('NMS threshold', 0.0, 1.0, 0.5, 0.05)

    model_architecture_col, input_size_col, classes_col = st.columns([2, 2, 6])
    model_architecture = model_architecture_col.selectbox('Scaled-YOLOv4 model', ('csp', 'p5', 'p6', 'p7'), index=1)
    input_size = input_size_col.selectbox('Model input size', (512, 640, 896, 1280, 1536), index=2)
    classes = classes_col.multiselect('Classes to display', COCO_CLASSES, ['person'])

    color_hex_col, font_size_col, _ = st.columns([2, 6, 2])
    color_hex = color_hex_col.color_picker('Color for bbox', '#0000ff')
    bbox_color = hex_to_bgr(color_hex)
    font_size = font_size_col.slider('Font size', 0.5, 1.5, 1.0, 0.1)

    od = initialize_od(model_architecture, input_size)

    num_cols_col, _, images_folder_col, _ = st.columns([3, 1, 12, 4])
    num_cols = num_cols_col.slider('Num of columns', 1, 5, 2, 1)
    images_folder = images_folder_col.text_input('Folder path containing images')
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

                    detections = process_image_cache(od, image, confidence_threshold, nms_threshold, classes=classes)
                    img = annotate_image_cache(image, detections, color=bbox_color, font_size=font_size)

                    all_images.append(img)

            for i in range(0, len(all_images), num_cols):
                cols = st.columns(num_cols)
                images = all_images[i:i+num_cols]
                for j, _ in enumerate(images):
                    cols[j].image(images[j], channels='BGR')


if __name__ == '__main__':
    main()
