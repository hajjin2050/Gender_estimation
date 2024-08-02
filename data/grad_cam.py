import numpy as np
import cv2
from PIL import Image
import torch

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) 구현 클래스.

    Parameters
    ----------
    model : torch.nn.Module
        Grad-CAM을 적용할 모델.
    target_layer : torch.nn.Module
        Grad-CAM을 적용할 대상 레이어.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        GradCAM 초기화 메서드.

        [input]
        model : torch.nn.Module
            Grad-CAM을 적용할 모델.
        target_layer : torch.nn.Module
            Grad-CAM을 적용할 대상 레이어.

        [output]
        None
        -------
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks()

    def hooks(self) -> None:
        """
        모델의 forward 및 backward 훅을 설정합니다.

        [input]
        None

        [output]
        None
        -------
        """   
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        입력 이미지에 대한 클래스 활성화 맵을 생성합니다.

        [input]
        input_image : torch.Tensor
            모델에 입력될 이미지 텐서.
        target_class : int, optional
            활성화 맵을 생성할 대상 클래스. 기본값은 None이며, 이 경우 예측된 클래스를 사용.

        [output]
        np.ndarray
            생성된 클래스 활성화 맵.
        -------
        """
        self.model.eval()
        input_image = input_image.to(next(self.model.parameters()).device)

        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax().item()
        
        target_class = min(target_class, output.size(1) - 1)

        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = np.uint8(cam * 255)

        return cam

def overlay_heatmap_on_image(img: np.ndarray, cam: np.ndarray) -> Image.Image:
    """
    원본 이미지에 Grad-CAM 히트맵을 오버레이합니다.

    [input]
    img : np.ndarray
        원본 이미지 배열.
    cam : np.ndarray
        Grad-CAM 히트맵 배열.

    [output]
    Image.Image
        히트맵이 오버레이된 이미지.
    -------
    """
    height, width, _ = img.shape

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = Image.fromarray(heatmap)
    base = Image.fromarray((img * 255).astype(np.uint8))

    combined = Image.blend(base.convert('RGBA'), overlay.convert('RGBA'), alpha=0.5)
    return combined