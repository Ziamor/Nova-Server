# Use an NVIDIA CUDA base image
ARG CUDA_IMAGE="12.1.1-devel-ubuntu22.04" 
FROM nvidia/cuda:${CUDA_IMAGE}

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install depencencies
RUN python3 -m pip install --upgrade pip cmake && \
	CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
	
WORKDIR /app

# Copy the rest of the application
COPY . /app/

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "-u", "llm-processing.py"]