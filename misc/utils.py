import cv2
import streamlit as st


@st.cache
def process_image_cache(od, image, confidence_threshold, nms_threshold):
    od.thresh = confidence_threshold
    od.nms_thresh = nms_threshold
    return od.detect_get_box_in([image], box_format='ltrb', classes=None)[0]


@st.cache
def annotate_image_cache(image, detections):
    draw_frame = image.copy()
    for det in detections:
        bb, score, det_class = det
        l, t, r, b = bb
        cv2.rectangle(draw_frame, (l, t), (r, b), (255, 0, 0), 2)
        label = f'{det_class}: {score:.2f}'
        cv2.putText(draw_frame, label, (l+5, t+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return draw_frame


def process_image(od, image, confidence_threshold, nms_threshold):
    od.thresh = confidence_threshold
    od.nms_thresh = nms_threshold
    return od.detect_get_box_in([image], box_format='ltrb', classes=None)[0]


def annotate_image(image, detections):
    draw_frame = image.copy()
    for det in detections:
        bb, score, det_class = det
        l, t, r, b = bb
        cv2.rectangle(draw_frame, (l, t), (r, b), (255, 0, 0), 2)
        label = f'{det_class}: {score:.2f}'
        cv2.putText(draw_frame, label, (l+5, t+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return draw_frame
