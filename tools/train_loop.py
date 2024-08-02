import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import io

def train_one_epoch(model: torch.nn.Module, device: torch.device, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer) -> tuple:
    """
    epoch 단위 모델 학습.
    ----------
    [input]
    model : torch.nn.Module
        학습할 모델.
    device : torch.device
        모델 및 데이터를 배치할 디바이스 (CPU 또는 GPU).
    train_loader : torch.utils.data.DataLoader
        학습 데이터를 제공하는 데이터 로더.
    optimizer : torch.optim.Optimizer
        모델의 매개변수를 업데이트하는 옵티마이저.

    [output]
    tuple
       train_loss, train_prec, train_recall, train_f1score.
    -------
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    return avg_loss, precision, recall, f1

def validate(model: torch.nn.Module, device: torch.device, val_loader: torch.utils.data.DataLoader) -> tuple:
    """
    검증 데이터셋으로 모델을 평가.
    ----------
    [input]
    model : torch.nn.Module
        평가할 모델.
    device : torch.device
        모델 및 데이터를 배치할 디바이스 (CPU 또는 GPU).
    val_loader : torch.utils.data.DataLoader
        검증 데이터를 제공하는 데이터 로더.

    [output]
    tuple
        val_loss, val_prec, val_recall, val_f1score, Confusion_matrix.
    
    -------

    """    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    cm = confusion_matrix(all_targets, all_preds)

    return avg_loss, precision, recall, f1, cm