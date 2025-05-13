# 使用 Nvidia CUDA 運行時映像檔 (Ubuntu 20.04 基礎)
FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

# 避免在 apt-get install 期間出現交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安裝 PPA 工具和基礎工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 添加 deadsnakes PPA 以獲取 Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa

# 安裝 Python 3.10, OpenJDK 17 和其他必要工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv openjdk-17-jdk git \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python -m ensurepip --upgrade # Use ensurepip to install/upgrade pip for python3.10

# 安裝 Spark
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -o /tmp/spark.tgz https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xzf /tmp/spark.tgz -C /opt/ \
    && mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME} \
    && rm /tmp/spark.tgz

# 設置工作目錄
WORKDIR /app

# 複製 requirements 文件
COPY requirements.txt .

# 從 requirements.txt 中移除 torch 和 torchvision，過濾後安裝其他依賴
RUN grep -vE '^torch|^torchvision' requirements.txt > requirements.filtered.txt || true
# 首先安裝過濾後的依賴 (確保 matplotlib, pandas, numpy<2.0 等被安裝)
RUN if [ -s requirements.filtered.txt ]; then python -m pip install --no-cache-dir -r requirements.filtered.txt; fi

# 安裝特定 GPU 版本的 PyTorch 和 Torchvision (匹配 CUDA 11.7)
RUN python -m pip install --no-cache-dir torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 清理臨時文件
RUN rm -f requirements.filtered.txt

# 複製應用代碼
COPY . .

# 設置環境變量
ENV PYTHONPATH=/app
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin
# Spark 的環境變量最好在 compose 文件中根據服務角色設置

# 暴露端口 (可選)
EXPOSE 7077 8080 9999

# 默認命令 (可被 docker-compose 覆蓋)
CMD ["bash"]

FROM bitnami/spark:3.5.0

USER root

# 安裝 Python 和必要的包
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 依賴
RUN pip3 install \
    torch \
    torchvision \
    numpy \
    pandas \
    matplotlib \
    scipy \
    scikit-learn \
    diffprivlib \
    psutil \
    tqdm \
    jupyter \
    notebook \
    py4j \
    pyspark \
    seaborn

# 設置工作目錄
WORKDIR /app

# 複製代碼
COPY . /app/

# 設置環境變量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib

# 不要切換用戶，保持 root
# USER 1001 