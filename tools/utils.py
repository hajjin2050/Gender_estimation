import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import mlflow

from data.grad_cam import GradCAM, overlay_heatmap_on_image

# Confusioin Matrix 관련 함수
def save_confusion_matrix(cm_buf: np.ndarray, run_dir: str, epoch: int) -> None:
    """
    Confusion Matrix 생성, MLflow에 저장.
    ----------
    [input]
    cm_buf : np.ndarray
        혼동 행렬 데이터.
    run_dir : str
        결과를 저장할 디렉토리 경로.
    epoch : int
        현재 에포크 번호.

    [output]
    None
    -------
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_buf, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_filename = f"confusion_matrix_epoch_{epoch+1}.png"
    cm_filepath = os.path.join(run_dir, cm_filename)
    plt.savefig(cm_filepath)
    plt.close()
    image = Image.open(cm_filepath)
    mlflow.log_image(image, cm_filename)
    return cm_filepath

# MLflow 로깅 함수
def log_mlflow_params(config: dict) -> None:
    """
    모델 파라미터 MLflow logging.
    ----------
    [input]
    config : dict
        학습에 사용된 구성 딕셔너리.

    [output]
        None
    -------
    """
    params = {
        "learning_rate": config["lr"],
        "epochs": config["epochs"],
        "drop_rate": config["drop_rate"],
        "batch_size": config["batch_size"],
        "weight_decay": config["weight_decay"],
        "Dataset": config["Dataset"],
        "Models": config["model_name"]
    }
    mlflow.log_params(params)

# MLflow 로깅 함수

def log_mlflow_metrics(metrics: dict, step: int) -> None:
    """
    train,validation 학습 지표 값 MLflow logging.
    ----------
    [input]
    metrics : dict
        학습 및 검증 평가지표 dict.
    step : int
        현재 단계 (epoch) 번호.

    [output]
        None
    -------
    """  
    mlflow.log_metrics(metrics, step=step)

#  MLflow 로깅 함수
def log_mlflow_images(inputs: torch.Tensor, model: torch.nn.Module, grad_cam: GradCAM, run_dir: str, epoch: int) -> None:
    """
    GradCAM 히트맵 이미지  MLflow logging.
    ----------
    [input]
    inputs : torch.Tensor
        입력 이미지 텐서.
    model : torch.nn.Module
        모델 객체.
    grad_cam : GradCAM
        GradCAM 객체.
    run_dir : str
        결과를 저장할 디렉토리 경로.
    epoch : int
        현재 에포크 번호.

    [output]
    None
    -------
    """
    cam = grad_cam.generate_cam(inputs)
    img = inputs[0].cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    combined = overlay_heatmap_on_image(img, cam)
    heatmap_filename = f'heatmap_overlay_epoch_{epoch+1}.png'
    heatmap_filepath = os.path.join(run_dir, heatmap_filename)
    combined.save(heatmap_filepath)
    mlflow.log_image(combined, heatmap_filename)