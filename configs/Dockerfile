FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        git \
        vim \
        wget \
        build-essential

COPY requirements.txt /tmp/

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

WORKDIR /workspace



