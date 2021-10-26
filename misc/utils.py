import cv2
import streamlit as st


COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush']


@st.cache
def process_image_cache(od, image, confidence_threshold, nms_threshold, classes=None):
    od.thresh = confidence_threshold
    od.nms_thresh = nms_threshold
    return od.detect_get_box_in([image], box_format='ltrb', classes=classes)[0]


@st.cache
def annotate_image_cache(image, detections, font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0),
                         font_size=1):
    draw_frame = image.copy()
    frame_h, frame_w, _ = image.shape
    for det in detections:
        bb, score, det_class = det
        l, t, r, b = bb
        l = max(l, 0)
        t = max(t, 0)
        r = min(r, frame_w)
        b = min(b, frame_h)
        cv2.rectangle(draw_frame, (l, t), (r, b), color, 2)
        label = f'{det_class}: {score:.2f}'
        cv2.putText(draw_frame, label, (l+5, t+30), font, font_size, color, 2)

    return draw_frame


def process_image(od, image, confidence_threshold, nms_threshold, classes=None):
    od.thresh = confidence_threshold
    od.nms_thresh = nms_threshold
    return od.detect_get_box_in([image], box_format='ltrb', classes=classes)[0]


def annotate_image(image, detections, font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0),
                   font_size=1):
    draw_frame = image.copy()
    frame_h, frame_w, _ = image.shape
    for det in detections:
        bb, score, det_class = det
        l, t, r, b = bb
        l = max(l, 0)
        t = max(t, 0)
        r = min(r, frame_w)
        b = min(b, frame_h)
        cv2.rectangle(draw_frame, (l, t), (r, b), color, 2)
        label = f'{det_class}: {score:.2f}'
        cv2.putText(draw_frame, label, (l+5, t+30), font, font_size, color, 2)

    return draw_frame


def process_tracks(od, tracker, image, confidence_threshold, nms_threshold, classes=None):
    od.thresh = confidence_threshold
    od.nms_thresh = nms_threshold
    detections = od.detect_get_box_in([image], box_format='ltwh', classes=classes)[0]
    return tracker.update_tracks(frame=image, raw_detections=detections)


def annotate_tracks(image, tracks, color=(255, 0, 0), font_size=1):
    draw_frame = image.copy()
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        _draw_track(draw_frame, track, color=color, font_size=font_size)
    return draw_frame


def _draw_track(frame, track, font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), font_size=1):
    frame_h, frame_w, _ = frame.shape
    l, t, r, b = [int(x) for x in track.to_ltrb(orig=True)]
    l = max(l, 0)
    t = max(t, 0)
    r = min(r, frame_w)
    b = min(b, frame_h)

    label = f'{track.get_det_class()} {track.track_id}: {track.get_det_conf():.2f}'
    cv2.rectangle(frame, (l, t), (r, b), color, 2)
    cv2.putText(frame, label, (l+5, t+30), font, font_size, color, 2)
