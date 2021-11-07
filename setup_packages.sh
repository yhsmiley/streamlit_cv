#!/usr/bin/env bash

# if you confirm no need to edit the packages (just use it)
pip3 install /media/data/ScaledYOLOv4 --no-binary=:all:
pip3 install /media/data/deep_sort_realtime
pip3 install /media/data/bytetrack_realtime --no-binary=:all:

# install packages as editable
# pip3 install -e /media/data/ScaledYOLOv4
# pip3 install -e /media/data/deep_sort_realtime
# pip3 install -e /media/data/bytetrack_realtime
