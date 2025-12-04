# Use Ubuntu as the base image
FROM ubuntu:22.04

ARG UID=1000
ARG GID=1000

# Set default shell during Docker image build to bash
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && \
    apt-get install -y software-properties-common wget && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        build-essential \
        ffmpeg \
        libsm6 \
        libxext6 \
        tesseract-ocr \
        libtesseract-dev \
        python3.13 \
        python3-pip \
        python3-tk \
        git \
        unzip \
        cmake \
        sudo && \
    rm -rf /var/lib/apt/lists/*

# Create a user and group with the same UID and GID as the host user
RUN groupadd -g $GID -o DSP \
    && useradd -u $UID -m -g DSP -G plugdev,sudo DSP

# Set no password for sudo for user DSP
RUN echo "DSP ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/dsp && \
    chmod 0440 /etc/sudoers.d/dsp

COPY . /DSP

WORKDIR /DSP/

RUN chown $UID:$GID /DSP

# Switch to DSP user as the final step
USER DSP

# Download and unzip the fonts
RUN wget -O /tmp/fonts.zip \
    "https://github.com/keshikan/DSEG/releases/download/v0.50beta1/fonts-DSEG_v050b1.zip" \
    && unzip /tmp/fonts.zip -d /tmp/fonts && rm /tmp/fonts.zip

# create the folder for fonts and copy the fonts to there then delete the fonts folder
RUN mkdir -p /tmp/ttf_collection && \
    find /tmp/fonts -type f -name "DSEG*.ttf" -exec cp {} /tmp/ttf_collection \; && \
    rm -f /tmp/ttf_collection/DSEG7SEGGCHAN* /tmp/ttf_collection/DSEGWeather* && \
    find /tmp/fonts -type f -name "DSEG*.ttf" -exec rm {} \; && \
    rm -rf /tmp/fonts

# Install TensorFlow with GPU support first (it will handle CUDA dependencies)
RUN pip install tensorflow[and-cuda]

# Install other Python dependencies
RUN pip install -r scripts/requirements.txt

RUN wget -qO- https://astral.sh/uv/install.sh | sh
ENV PATH="/home/DSP/.local/bin:${PATH}"

# Allow PFE to run without restrictions within the container
RUN git config --global --add safe.directory '*'