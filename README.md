# Streamlit for CV applications

This repo contains some CV applications using streamlit.

## Installation

- Build docker image (uncomment the required packages, if not can be installed at the next step)
- Install required packages (choose if packages should be installed as editable)
```bash
./setup_packages.sh
```

## Usage

- Run ScaledYOLOv4 on a single image
```bash
streamlit run app_single_image.py
```
- Run ScaledYOLOv4 on a folder of images
```bash
streamlit run app_folder_images.py
```
- Run ScaledYOLOv4 (with/without tracking) on a video file
```bash
streamlit run app_videofile.py
```
- Run ScaledYOLOv4 on webcam / video file / rtsp stream with streamlit-webrtc (buggy, use at your own risk)
```bash
streamlit run app_video_webrtc.py
```
