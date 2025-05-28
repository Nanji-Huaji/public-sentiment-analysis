FROM ubuntu:22.04

WORKDIR /app

# 安装基本依赖
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制本地的Anaconda安装脚本
COPY Anaconda3-2024.10-1-Linux-x86_64.sh /tmp/anaconda.sh

# 安装Anaconda
RUN bash /tmp/anaconda.sh -b -p /opt/anaconda && \
    rm /tmp/anaconda.sh

# 将anaconda添加到PATH
ENV PATH="/opt/anaconda/bin:${PATH}"

# 初始化conda
RUN conda init bash

# 复制项目文件
COPY . /app/

# 创建nlp环境 (Python 3.10)
RUN conda create -n nlp python=3.10 -y && \
    conda run -n nlp pip install \
    torch torchvision torchaudio \
    transformers \
    datasets \
    pandas \
    scikit-learn \
    matplotlib \
    tensorboard

# 创建media_crawler环境 (Python 3.12)
RUN conda create -n media_crawler python=3.12 -y && \
    conda run -n media_crawler pip install \
    torch torchvision torchaudio \
    transformers \
    datasets \
    pandas \
    scikit-learn

# 设置shell初始化脚本
RUN echo 'alias activate_nlp="conda activate nlp"' >> ~/.bashrc && \
    echo 'alias activate_media_crawler="conda activate media_crawler"' >> ~/.bashrc

# 默认激活nlp环境
RUN echo "conda activate nlp" >> ~/.bashrc

# 设置默认命令
CMD ["bash"]