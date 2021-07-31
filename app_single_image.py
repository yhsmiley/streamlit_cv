import cv2
import numpy as np
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
    labels = []
    for det in detections:
        bb, score, det_class = det 
        l,t,r,b = bb
        cv2.rectangle(draw_frame, (l,t), (r,b), (255,0,0), 2)
        label = f"{det_class}: {score:.2f}"
        cv2.putText(draw_frame, label, (l+5, t+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        labels.append(label)
    
    return draw_frame, labels


st.set_page_config(layout="wide")
st.title("Object detection with ScaledYOLOv4")
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
nms_threshold = st.slider("NMS threshold", 0.0, 1.0, 0.5, 0.05)

od = initialize_od()

if img_file_buffer is not None:
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    detections = process_image(od, image, confidence_threshold, nms_threshold)
    image, labels = annotate_image(image, detections)

    st.image(
        image, caption=f"Processed image", use_column_width=True, channels="BGR"
    )

    st.write(labels)
