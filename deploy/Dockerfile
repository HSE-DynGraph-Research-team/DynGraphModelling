# syntax=docker/dockerfile:1

# FROM python:3.8-slim-buster
FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04 as intermediate


RUN apt-get update -qq && \
    apt-get install -y git wget gcc build-essential && \
    apt-get clean

# install conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda && /opt/conda/bin/conda create -y -n myenv python=3.8



ARG SSH_PRIVATE_KEY
RUN mkdir /root/.ssh/ \
    && echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa \
    && chmod 600 /root/.ssh/id_rsa \
    && touch /root/.ssh/known_hosts \
    && ssh-keyscan github.com >> /root/.ssh/known_hosts 

RUN git clone git@github.com:HSE-DynGraph-Research-team/DynGraph-modelling.git

# RUN echo "deb http://repo.yandex.ru/yandex-disk/deb/ stable main" | tee -a /etc/apt/sources.list.d/yandex-disk.list > /dev/null && \
#  wget http://repo.yandex.ru/yandex-disk/YANDEX-DISK-KEY.GPG -O- | apt-key add - && \ 
#  apt-get update && \ 
#  apt-get install -y yandex-disk

WORKDIR /DynGraph-modelling
RUN mkdir /DynGraph-modelling/data

# COPY requirements.txt /app/DynGraph-modelling
ENV PATH=/opt/conda/envs/myenv/bin:$PATH    
RUN pip install torch==1.8.0 \
    && pip install --no-cache-dir torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html \
    && pip install --no-cache-dir torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html \
    && pip install --no-cache-dir torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html \
    && pip install --no-cache-dir torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html \
    && pip install --no-cache-dir torch-geometric==1.6.3 \
    && pip install --no-cache-dir torch-geometric-temporal==0.25  \
    && pip install $(cat requirements.txt | grep -v '^torch')

RUN pip install gdown && \
    gdown https://drive.google.com/uc?id=1CIHXCEA_7UT9rnlw4P8J8gGN6hYI1l00 -O ./DynGraph-modelling/data/wikipedia.csv && \ 
    gdown https://drive.google.com/uc?id=1Lby2Sf9KpmaueZrE_CnSVDzQCDCKR4Ye -O ./DynGraph-modelling/data/reddit.csv && \ 
    gdown https://drive.google.com/uc?id=198CrBxdWI_6LVBIqmK7jKkCPnUQ6_Xdb -O ./DynGraph-modelling/data/eth_data_300k.csv

RUN pip uninstall -y pip


# FROM ubuntu:18.04
# FROM nvcr.io/pytorch:20.12-py3
# FROM nvidia/cuda:11.2.1-runtime
FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04 as runner
COPY --from=intermediate /DynGraph-modelling /app/DynGraph-modelling
COPY --from=intermediate /DynGraph-modelling/requirements.txt /app/DynGraph-modelling

COPY --from=intermediate /opt/conda/envs/myenv/bin /opt/conda/envs/myenv/bin
COPY --from=intermediate /opt/conda/envs/myenv/lib /opt/conda/envs/myenv/lib

# do some env settings
WORKDIR /app/DynGraph-modelling/
ENV PATH=/opt/conda/envs/myenv/bin:$PATH
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD [ "python",  "scenarios.py"]
