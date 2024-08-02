FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime 

RUN apt-get update
RUN apt-get install git
RUN pip install -y libg1-mesa-glx libglib2.0-0 mlflow==2.14.3 numpy==2.0.1 opencv_python==4.10.0.84 pandas==2.2.2 pillow==10.4.0 timm==1.0.8 ray==2.33.0 scikit_learn==1.5.1 seaborn==0.13.2

WORKDIR /workspace

RUN git clone https://github.com/hajjin2050/Gender_estimation.git .