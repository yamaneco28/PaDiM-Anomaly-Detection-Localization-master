FROM nvcr.io/nvidia/l4t-base:r32.5.0

# skip timezone setting
ENV DEBIAN_FRONTEND=noninteractive

# Force color prompt in terminal
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

# japanise setting
RUN apt-get update
RUN apt-get install -y locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

# install utility
RUN apt-get update && \
    apt-get install -y \
    wget \
    curl \
    git \
    vim \
    lsb-release \
    gnupg

# python setting
RUN apt-get update && apt-get install -y python3.7 python3-pip
RUN pip3 install --upgrade pip
RUN apt-get install -y python3-matplotlib
WORKDIR /root/requirements
ADD requirements.txt /root/requirements
RUN pip3 install -r requirements.txt

# install pytorch
RUN wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
RUN apt-get install -y libopenblas-base libopenmpi-dev
RUN pip3 install Cython
RUN pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl
# RUN pip3 install scikit-image

# torchvision
# RUN apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
# RUN export BUILD_VERSION=0.9.0
# RUN git clone --branch $BUILD_VERSION https://github.com/pytorch/vision torchvision
# RUN cd torchvision
# RUN python3 setup.py install --user
# RUN cd ../  # attempting to load torchvision from build dir will result in import error
# RUN pip3 install 'pillow<7' # always needed for Python 2.7, not needed torchvision v0.5.0+ with Python 3.6

# work directory setting
WORKDIR /root/workspace
