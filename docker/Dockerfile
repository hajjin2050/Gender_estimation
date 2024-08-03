# 베이스 이미지 설정
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
# Create a temporary directory with appropriate permissions
RUN mkdir -p /tmp/apt && chmod 1777 /tmp/apt
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Update the system packages and install necessary packages
RUN apt-get -o Dir::Cache::Archives="/tmp/apt" -y update && \
    apt-get -o Dir::Cache::Archives="/tmp/apt" install -y git libgl1-mesa-glx libglib2.0-0

# 필요한 Python 패키지 설치
RUN pip install mlflow==2.14.3 numpy opencv_python panda pillow timm ray scikit_learn==1.5.1 seaborn==0.13.2 openpyxl

# 작업 디렉토리 설정
WORKDIR /workspace
COPY ./* /workspace

# MLflow 서버 시작 스크립트 추가
RUN chmod +x /workspace/start_mlflow_server.sh

# CMD 명령어 설정: MLflow 서버를 백그라운드에서 실행하고 bash 쉘을 시작
CMD ["/bin/bash", "-c", "/workspace/start_mlflow_server.sh & bash"]