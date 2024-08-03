import os

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 데이터 로더 함수
def get_data_loaders(data_dir: str, batch_size: int = 32, input_size: tuple = (224, 224)) -> tuple:
    """
    사전 준비된 train, val 파일을 통해 데이터 로더 정의.
    ----------
    [input]
    data_dir : str
        데이터 셋 경로.
    batch_size : int, optional
        배치 사이즈. 기본값은 32.
    input_size : tuple, optional
        이미지 크기 조정 크기. 기본값은 (224, 224).
    [output]
    tuple
        train_loader와 val_loader를 포함하는 튜플.
    -------
    """
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    val_dataset = ImageFolder(os.path.join(data_dir, 'valid'), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

