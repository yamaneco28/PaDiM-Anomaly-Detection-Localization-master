# FROM ubuntu:20.04
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
    python3-pip \
    lsb-release \
    gnupg

# install python package
RUN apt-get install -y --no-install-recommends python3-matplotlib
RUN pip3 install --upgrade pip
WORKDIR /root/requirements
ADD requirements.txt /root/requirements
RUN pip3 install -r requirements.txt

WORKDIR /root/workspace
