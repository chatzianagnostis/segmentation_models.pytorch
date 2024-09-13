import torchvision
import torch.nn as nn
from ._base import EncoderMixin

class MobileNetV3Encoder(nn.Module, EncoderMixin):
    def __init__(self, out_channels, depth=5, model_type="large", **kwargs):
        super().__init__()

        # Choose between mobilenet_v3_large or mobilenet_v3_small
        if model_type == "large":
            base_model = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
        else:
            base_model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
        
        # Use the pre-trained features from the model
        self.features = base_model.features
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        
        # Remove the classifier, we only need the feature extractor
        del base_model.classifier

    def get_stages(self):
        # This method splits the feature extractor into stages
        return [
            nn.Identity(),
            self.features[:2],
            self.features[2:4],
            self.features[4:7],
            self.features[7:10],
            self.features[10:],  # Adjust the depth according to your requirement
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features


    def load_state_dict(self, state_dict, **kwargs):
        # Filter out classifier keys from the state dict
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
        super().load_state_dict(state_dict, **kwargs)

# Define encoders with their pretrained settings
mobilenet_encoders = {
    "mobilenet_v3_large": {
        "encoder": MobileNetV3Encoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": {
            "out_channels": (3, 16, 24, 40, 80, 960),  # Update based on feature shapes
        },
    },
    "mobilenet_v3_small": {
        "encoder": MobileNetV3Encoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": {
            "out_channels": (3, 16, 24, 40, 48, 576),  # Update based on small variant
        },
    },
}
