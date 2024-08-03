# 베이스 이미지 설정
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# Create a temporary directory with appropriate permissions
RUN mkdir -p /tmp/apt && chmod 1777 /tmp/apt

# Update the system packages and install necessary packages
RUN apt-get -o Dir::Cache::Archives="/tmp/apt" -y update && \
    apt-get -o Dir::Cache::Archives="/tmp/apt" install -y git libgl1-mesa-glx libglib2.0-0

# 필요한 Python 패키지 설치
COPY requirements.txt .

# 작업 디렉토리 설정
WORKDIR /workspace

# 모든 파일을 /workspace로 복사
COPY . /workspace

# MLflow 서버 시작 스크립트 추가

WORKDIR /workspace
RUN chmod +x /workspace/start_mlflow_server.sh
RUN pip install -r requirements.txt

# CMD 명령어 설정: MLflow 서버를 백그라운드에서 실행하고 bash 쉘을 시작
CMD ["/bin/bash", "-c", "/workspace/start_mlflow_server.sh & bash"]
