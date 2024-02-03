# Based on
#   https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html#using-docker-with-pytorch-pre-installed
#   https://pytorch.org/hub/ultralytics_yolov5/

FROM ubuntu:jammy AS base

RUN apt-get -y update && apt-get -y upgrade

RUN apt-get -y install curl sudo
RUN apt-get -y install libusb-1.0-0-dev
RUN apt-get -y install python3-pip

RUN pip install -U pip
RUN pip install 'tensorflow-cpu==2.13.0'
RUN pip install 'torch==2.2.0' torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install 'gitpython>=3.1.30'
RUN pip install 'Pillow>=10.0.1'
RUN pip install 'numpy>=1.23.5'
RUN pip install 'scipy>=1.11.4'
RUN pip install 'onnx>=1.12.0'
RUN pip install 'onnxruntime' 'onnxruntime-gpu'
RUN pip install 'openvino-dev>=2023.0'
RUN pip install 'super_gradients'
RUN pip install 'nvidia-pyindex'
RUN pip install 'onnx-graphsurgeon'
RUN pip install 'sng4onnx'
RUN pip install 'tflite_support'
RUN pip install 'onnx2tf>=1.15.4,<=1.17.5'
RUN pip install 'nncf>=2.5.0'
RUN pip install 'ultralytics'

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN apt-get -y update && apt-get -y install edgetpu-compiler	

FROM base AS data

WORKDIR /opt/build

RUN pip list --format freeze > pip.freeze.txt

ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt .
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt .
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt .
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-oiv7.pt .
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-oiv7.pt .
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-oiv7.pt .
# The "l" or "x" models take too much space so we ignore them here
#ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt .
#ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-oiv7.pt .
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/calibration_image_sample_data_20x128x128x3_float32.npy.zip .

COPY coco-labels.txt .
COPY oiv7-labels.txt .
COPY oiv7 oiv7

FROM data AS process

WORKDIR /opt/build
RUN mkdir yolov8

# Make aggregated labels
COPY build_labels.py .
RUN python3 build_labels.py

COPY build_onnx.py .
RUN python3 build_onnx.py

COPY build_openvino.py .
RUN python3 build_openvino.py

COPY build_edgetpu.py .
RUN python3 build_edgetpu.py

COPY LICENSE README.md yolov8/

FROM scratch

COPY --from=process /opt/build/yolov8 /


