FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安裝基本依賴
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安裝 pip
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py

# 設置 Python 3.10 為默認版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 設置 JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# 安裝 Spark
ENV SPARK_VERSION=3.4.1
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME} \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# 設置 PATH
ENV PATH=$PATH:${SPARK_HOME}/bin:${SPARK_HOME}/sbin

# 設置 Spark GPU 相關環境變量
ENV SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=1 -Dspark.worker.resource.gpu.discoveryScript=/opt/spark/examples/src/main/scripts/getGpusResources.sh"
ENV SPARK_EXECUTOR_OPTS="-Dspark.executor.resource.gpu.amount=1"

# 安裝 Python 依賴
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# 創建工作目錄
WORKDIR /app

# 複製啟動腳本
COPY spark-entrypoint.sh /
RUN chmod +x /spark-entrypoint.sh

# 設置入口點
ENTRYPOINT ["/spark-entrypoint.sh"] 