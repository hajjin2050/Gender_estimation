import sys
import os
import json
import argparse
import importlib

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd

sys.path.append("/workspace")

#모델 클래스 로드 함수
def load_model_class(model_name: str) -> type:
    """
    모델 클래스 로드
    -------------
    input : model_name(Config['model_name']) 
    output : model_class(timm활용)
    --------------
    """
    module_name = f"model.{model_name}"
    class_name = model_name
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class

#모델 로드 함수
def load_model(model_path: str, model_class: type, device: torch.device) -> torch.nn.Module:
    """
    모델 로드
    -------------
    input : 
        model_path: str, 모델 파일 경로
        model_class: type, 모델 클래스
        device: torch.device, 모델을 로드할 디바이스
    output : 모델 객체
    --------------
    """
    model = model_class(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    model.to(device)
    model.eval()
    return model

# 이미지 전처리 함수
def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Resize, Normalize, ToTensor 적용
    -------------
    input : image_path(str) 이미지 파일 경로
    output : 전처리된 이미지 텐서
    --------------
    """
    input_size = (224, 224)
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)
    return image

# 모델 추론 함수
def predict(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    이미지(gt없음) 추론
    -------------
    input : 
        model: torch.nn.Module, 학습된 모델
        image_tensor: torch.Tensor, 전처리된 이미지 텐서
        device: torch.device, 디바이스
    output : 예측 결과
    --------------
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities.cpu().numpy()

#메인 함수
def main(config_path: str, model_path: str, input_path: str):
    """
    테스트에 필요한 구성  파일 경로 입력
    -------------
    input : 
        config_path: str, 구성 파일 경로
        model_path: str, 모델 파일 경로
        input_path: str, 입력 이미지 파일 또는 디렉토리 경로
    output : None
    --------------
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_class = load_model_class(config["model_name"])
    model = load_model(model_path, model_class, device)

    if os.path.isdir(input_path):
        image_paths = [os.path.join(input_path, fname) for fname in os.listdir(input_path) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
    else:
        image_paths = [input_path]

    results = {
        "Input Image": [],
        "Predicted Gender": [],
        "Confidence": []
    }

    classes = ["Male", "Female"]

    for image_path in image_paths:
        image_tensor = preprocess_image(image_path)
        probabilities = predict(model, image_tensor, device)
        predicted_class = classes[np.argmax(probabilities)]
        confidence = np.max(probabilities)

        results["Input Image"].append(image_path)
        results["Predicted Gender"].append(predicted_class)
        results["Confidence"].append(confidence)

    df = pd.DataFrame(results)
    
    model_folder_name = os.path.basename(os.path.dirname(model_path))
    output_excel = f"{model_folder_name}_inference_results.xlsx"
    df.to_excel(output_excel, index=False)

    print(f"Inference results saved to {output_excel}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with trained model")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image file or directory")
    args = parser.parse_args()
    main(args.config, args.model, args.input)