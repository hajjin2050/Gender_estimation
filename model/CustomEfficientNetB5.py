from model.base_model import BaseModel
import timm
import torch

class EfficientNetB5(BaseModel):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetB5, self).__init__()
        self.model = timm.create_model('efficientnet_b5', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class CustomEfficientNetB5(EfficientNetB5):
    def __init__(self, num_classes=2, pretrained=True, drop_rate=0.2):
        super(CustomEfficientNetB5, self).__init__(num_classes=num_classes, pretrained=pretrained)
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(drop_rate),
            torch.nn.Linear(self.model.classifier.in_features, num_classes)
        )
        self.grad_cam_layer = self.model.conv_head  # Grad-CAM에 사용할 레이어를 지정
