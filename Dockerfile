FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
COPY . /app/
# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -r /app/requirements.txt
RUN pip install protobuf



RUN if ! update-alternatives --query python > /dev/null 2>&1; then \
        update-alternatives --install /usr/bin/python python /usr/bin/python3 1; \
    fi


USER root
EXPOSE 7860
CMD ["python", "/app/app.py"]
