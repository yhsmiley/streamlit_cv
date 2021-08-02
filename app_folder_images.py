from pathlib import Path

import cv2
import streamlit as st

from ScaledYOLOv4.scaled_yolov4 import Scaled_YOLOV4


@st.cache(allow_output_mutation=True)
def initialize_od():
    yolov4 = Scaled_YOLOV4( 
        bgr=True,
        gpu_device=0,
        model_image_size=608,
        max_batch_size=1,
        half=True,
        same_size=True
    )

    return yolov4

@st.cache
def process_image(od, image, confidence_threshold, nms_threshold):
    od.thresh = confidence_threshold
    od.nms_thresh = nms_threshold
    detections = od.detect_get_box_in([image], box_format='ltrb', classes=None)[0]

    return detections

@st.cache
def annotate_image(image, detections):
    draw_frame = image.copy()
    for det in detections:
        bb, score, det_class = det 
        l,t,r,b = bb
        cv2.rectangle(draw_frame, (l,t), (r,b), (255,0,0), 2)
        label = f"{det_class}: {score:.2f}"
        cv2.putText(draw_frame, label, (l+5, t+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    
    return draw_frame


st.set_page_config(layout="wide")
st.title("Object detection with ScaledYOLOv4 (Folder of images)")
images_folder = st.text_input("Folder path containing images")
confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
nms_threshold = st.slider("NMS threshold", 0.0, 1.0, 0.5, 0.05)
num_cols = st.slider("Number of display columns", 1, 5, 2, 1)

od = initialize_od()

images_folder = str(images_folder) or None

suffixes = [".png", ".jpg", ".jpeg"]
if images_folder is not None:
    if not Path(images_folder).is_dir():
        st.error("Folder does not exist!")
    else:
        all_images = []
        for image_path in Path(images_folder).glob("*"):
            if image_path.suffix in suffixes:
                image = cv2.imread(str(image_path))

                detections = process_image(od, image, confidence_threshold, nms_threshold)
                img = annotate_image(image, detections)

                all_images.append(img)

        for i in range(0, len(all_images), num_cols):
            cols = st.beta_columns(num_cols)
            images = all_images[i:i+num_cols]
            for j in range(len(images)):
                cols[j].image(images[j], channels="BGR")
