FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

ADD ./Miniconda3-latest-Linux-x86_64.sh /
RUN mv /Miniconda3-latest-Linux-x86_64.sh ~/miniconda.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install conda-build && \
     /opt/conda/bin/conda install numpy pyyaml scipy ipython&& \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
RUN conda install pytorch torchvision cpuonly -c pytorch && /opt/conda/bin/conda clean -ya
RUN mkdir /code
ADD ./code /code
WORKDIR /code

