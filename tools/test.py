import sys
sys.path.append("/workspace")
import os
import json
import torch
import argparse
import importlib
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def load_model_class(model_name: str) -> type:
    """
    모델 모듈에서 모델 클래스를 로드
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

def preprocess_image(image_path: str) -> torch.Tensor:
    """
    이미지 전처리
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

def predict(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    모델을 사용하여 이미지를 예측함
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

def main(config_path: str, model_path: str, test_dir: str):
    """
    메인 함수
    -------------
    input : 
        config_path: str, 구성 파일 경로
        model_path: str, 모델 파일 경로
        test_dir: str, 테스트 데이터셋 디렉토리 경로
    output : None
    --------------
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_class = load_model_class(config["model_name"])
    model = load_model(model_path, model_class, device)

    classes = sorted(os.listdir(test_dir))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    results = {
        "Input Image": [],
        "GT": [],
        "Predicted": [],
        "Dataset":[]
    }

    gt_labels = []
    pred_labels = []

    for cls_name in classes:
        class_dir = os.path.join(test_dir, cls_name)
        image_paths = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]

        for image_path in image_paths:
            image_tensor = preprocess_image(image_path)
            probabilities = predict(model, image_tensor, device)
            predicted_class = classes[np.argmax(probabilities)]

            gt_labels.append(class_to_idx[cls_name])
            pred_labels.append(class_to_idx[predicted_class])

            results["Input Image"].append(image_path)
            results["GT"].append(cls_name)
            results["Predicted"].append(predicted_class)
            results["Dataset"].append(test_dir.split("/")[-1])
            
    precision, recall, f1, _ = precision_recall_fscore_support(gt_labels, pred_labels, average='weighted')

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    df = pd.DataFrame(results)
    model_folder = os.path.dirname(model_path)
    model_folder_name = os.path.basename(os.path.dirname(model_path))
    output_excel =os.path.join(model_folder, f"{model_folder_name}_test_results.xlsx")
    df.to_excel(output_excel, index=False)

    print(f"Test results saved to {output_excel}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model with a test dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the test dataset directory")
    args = parser.parse_args()
    main(args.config, args.model, args.test_dir)
