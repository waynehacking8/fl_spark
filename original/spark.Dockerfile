FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# 設置環境變量
ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV SPARK_VERSION=3.4.1
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$JAVA_HOME/bin

# 安裝基本依賴
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    openjdk-11-jdk \
    python3.9 \
    python3.9-distutils \
    python3.9-dev \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 設置 Python 3.9 為默認版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# 安裝 pip
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.9 get-pip.py \
    && rm get-pip.py

# 安裝 Python 依賴
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# 下載並安裝 Spark
RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# 設置 GPU 相關配置
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# 創建工作目錄
WORKDIR /app

# 複製啟動腳本
COPY spark-entrypoint.sh /spark-entrypoint.sh
RUN chmod +x /spark-entrypoint.sh

# 設置入口點
ENTRYPOINT ["/spark-entrypoint.sh"] 