FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git g++ r-base r-base-dev r-cran-data.table python3 python3-pip \
    ca-certificates cpp make libltdl-dev wget unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy pandas scipy

WORKDIR /CliPP
COPY setup.py run_clipp_main.py LICENSE README.md ./
COPY src/ ./src/
COPY sample/ ./sample/
RUN CLIPP_USE_CUDA=0 python3 setup.py build
