FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ENV OPENMPI_VERSION=4.0.5
ENV CUDA_HOME=/usr/local/cuda-10.1
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"

RUN apt-get update
RUN apt-get install -y wget libibverbs-dev libsysfs-dev cmake libboost-all-dev libeigen3-dev libbz2-dev liblzma-dev ssh git

# update OpenMPI to avoid horovod bug
RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-${OPENMPI_VERSION}.tar.gz &&\
    gunzip -c openmpi-${OPENMPI_VERSION}.tar.gz | tar xf - &&\
    cd openmpi-${OPENMPI_VERSION} &&\
    ./configure --prefix=/usr/local/mpi --enable-orterun-prefix-by-default \
        --with-verbs --disable-getpwuid &&\
    make -j$(nproc) all && make install &&\
    ldconfig &&\
    cd - && rm -r openmpi-${OPENMPI_VERSION} && rm openmpi-${OPENMPI_VERSION}.tar.gz

# basic python packages
RUN conda install cython
RUN pip install msgpack msgpack_numpy cytoolz more_itertools matplotlib scipy nltk transformers tensorboard \
    ipdb lz4 lmdb jsonlines gputil opencv-python pandas \
    'git+https://github.com/facebookresearch/detectron2.git' \
    xlrd scikit-learn git+https://github.com/PetrochukM/PyTorch-NLP.git pkuseg spacy ftfy

RUN python -m spacy download en_core_web_sm
RUN python -m spacy download zh_core_web_sm

# horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod && ldconfig

RUN git clone https://github.com/cocodataset/cocoapi.git /tmp/cocoapi
WORKDIR /tmp/cocoapi
RUN cd PythonAPI && python setup.py build_ext install

WORKDIR /src
