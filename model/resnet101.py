import timm
from torch import nn

class resnet101(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, drop_rate=0.2):
        super(resnet101, self).__init__()
        self.model = timm.create_model('resnet101', pretrained=pretrained, num_classes=num_classes)
        self.model.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.model.num_features, num_classes)
        )
        self.grad_cam_layer = self.model.features  # Grad-CAM에 사용할 레이어를 지정

    def forward(self, x):
        return self.model(x)