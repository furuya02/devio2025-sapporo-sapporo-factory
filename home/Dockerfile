FROM dustynv/l4t-pytorch:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_EXTRA_INDEX_URL="" \
    PIP_INDEX_URL="https://pypi.org/simple/" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Remove Jetson-specific pip configurations
RUN rm -f /etc/pip.conf /root/.pip/pip.conf ~/.pip/pip.conf 2>/dev/null || true

# Install system dependencies for GUI and OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # OpenGL/Mesa libraries
        libgl1-mesa-glx \
        libgl1-mesa-dri \
        # GUI libraries
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        libfontconfig1 \
        # Python GUI support
        python3-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Python Package Installation
# ============================================================================

# Core scientific computing packages
RUN pip install \
        --trusted-host pypi.org \
        --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org \
        --index-url https://pypi.org/simple/ \
        # Core dependencies (specific versions for compatibility)
        numpy==1.24.3 \
        pillow \
        # Computer vision
        opencv-python \
        # Machine learning
        ultralytics \
        # Scientific computing
        matplotlib \
        pandas \
        # Utilities
        pyyaml \
        gitpython

RUN pip install \
        psutil \
        seaborn \
        scipy        
RUN pip install Jetson.GPIO
RUN pip install boto3
RUN pip install thop
RUN pip install scikit-learn

WORKDIR /src
