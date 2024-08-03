# 베이스 이미지 설정
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime 

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y git libgl1-mesa-glx libglib2.0-0

# 필요한 Python 패키지 설치
RUN pip install mlflow==2.14.3 numpy==2.0.1 opencv_python==4.10.0.84 pandas==2.2.2 pillow==10.4.0 timm==1.0.8 ray==2.33.0 scikit_learn==1.5.1 seaborn==0.13.2 openpyxl

# 작업 디렉토리 설정
WORKDIR /workspace

# Git 리포지토리 클론 (리포지토리 URL을 실제 URL로 변경해야 합니다)
RUN git clone https://github.com/hajjin2050/Gender_estimation.git .

# MLflow 서버 시작 스크립트 추가
COPY ./start_mlflow_server.sh /workspace/start_mlflow_server.sh
RUN chmod +x /workspace/start_mlflow_server.sh

# CMD 명령어 설정: MLflow 서버를 백그라운드에서 실행하고 bash 쉘을 시작
CMD ["/bin/bash", "-c", "/workspace/start_mlflow_server.sh & bash"]