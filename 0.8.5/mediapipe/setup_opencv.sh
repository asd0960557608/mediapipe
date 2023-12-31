#!/bin/bash
# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
#
# Script to build OpenCV from source code and modify the MediaPipe opencv config.
# Note that this script only has been tested on Debian 9 and Ubuntu 16.04.
#
# To have a full installation:
# $ cd <mediapipe root dir>
# $ sh ./setup_opencv.sh
#
# To only modify the mediapipe config for opencv:
# $ cd <mediapipe root dir>
# $ sh ./setup_opencv.sh config_only

set -e
if [ "$1" ] && [ "$1" != "config_only" ]
  then
    echo "Unknown input argument. Do you mean \"config_only\"?"
    exit 0
fi

opencv_build_file="$( cd "$(dirname "$0")" ; pwd -P )"/third_party/opencv_linux.BUILD
workspace_file="$( cd "$(dirname "$0")" ; pwd -P )"/WORKSPACE

if [ -z "$1" ]
  then
    echo "Installing OpenCV from source"
    sudo apt update && sudo apt install build-essential git
    sudo apt install cmake ffmpeg libavformat-dev libdc1394-22-dev libgtk2.0-dev \
                     libjpeg-dev libpng-dev libswscale-dev libtbb2 libtbb-dev \
                     libtiff-dev
    rm -rf /tmp/build_opencv
    mkdir /tmp/build_opencv
    cd /tmp/build_opencv
    git clone https://github.com/opencv/opencv_contrib.git
    git clone https://github.com/opencv/opencv.git
    mkdir opencv/release
    cd opencv_contrib
    git checkout 4.5.5
    cd ../opencv
    git checkout 4.5.5
    cd release
    cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_opencv_ts=OFF \
          -DOPENCV_EXTRA_MODULES_PATH=~/build_opencv/opencv_contrib/modules \
          -DBUILD_opencv_aruco=OFF -DBUILD_opencv_bgsegm=OFF -DBUILD_opencv_bioinspired=OFF \
          -DBUILD_opencv_ccalib=OFF -DBUILD_opencv_datasets=OFF -DBUILD_opencv_dnn=OFF \
          -DBUILD_opencv_dnn_objdetect=OFF -DBUILD_opencv_dpm=OFF -DBUILD_opencv_face=OFF \
          -DBUILD_opencv_fuzzy=OFF -DBUILD_opencv_hfs=OFF -DBUILD_opencv_img_hash=OFF \
          -DBUILD_opencv_js=OFF -DBUILD_opencv_line_descriptor=OFF -DBUILD_opencv_phase_unwrapping=OFF \
          -DBUILD_opencv_plot=OFF -DBUILD_opencv_quality=OFF -DBUILD_opencv_reg=OFF \
          -DBUILD_opencv_rgbd=OFF -DBUILD_opencv_saliency=OFF -DBUILD_opencv_shape=OFF \
          -DBUILD_opencv_structured_light=OFF -DBUILD_opencv_surface_matching=OFF \
          -DBUILD_opencv_world=OFF -DBUILD_opencv_xobjdetect=OFF -DBUILD_opencv_xphoto=OFF \
          -DCV_ENABLE_INTRINSICS=ON -DWITH_EIGEN=ON -DWITH_PTHREADS=ON -DWITH_PTHREADS_PF=ON \
          -DWITH_JPEG=ON -DWITH_PNG=ON -DWITH_TIFF=ON \
          -D WITH_TBB=ON \
          -D WITH_CUDA=ON \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D CUDA_ARCH_BIN=8.6 \
          -D BUILD_opencv_cudacodec=OFF \
          -D ENABLE_FAST_MATH=1 \
          -D CUDA_FAST_MATH=1 \
          -D WITH_CUBLAS=1 \
          -D WITH_V4L=ON \
          -D WITH_QT=OFF \
          -D WITH_OPENGL=ON \
          -D WITH_GSTREAMER=ON \
          -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
          -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") ..
          #-DWITH_IPP=OFF -DBUILD_TBB=OFF \
          #-DBUILD_IPP_IW=OFF \
          #-DBUILD_WITH_DYNAMIC_IPP=OFF \
    make -j 16
    sudo make install
    #rm -rf /tmp/build_opencv
    echo "OpenCV has been built. You can find the header files and libraries in /usr/local/include/opencv2/ and /usr/local/lib"

    # https://github.com/cggos/dip_cvqt/issues/1#issuecomment-284103343
    sudo touch /etc/ld.so.conf.d/mp_opencv.conf
    sudo bash -c  "echo /usr/local/lib >> /etc/ld.so.conf.d/mp_opencv.conf"
    sudo ldconfig -v
fi

# Modify the build file.
echo "Modifying MediaPipe opencv config"

sed -i "s/lib\/x86_64-linux-gnu/lib/g" $opencv_build_file
linux_opencv_config=$(grep -n 'linux_opencv' $workspace_file | awk -F  ":" '{print $1}')
path_line=$((linux_opencv_config + 2))
sed -i "$path_line d" $workspace_file
sed -i "$path_line i\    path = \"/usr/local\"," $workspace_file
echo "Done"
