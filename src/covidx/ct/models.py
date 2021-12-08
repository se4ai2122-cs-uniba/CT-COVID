import torch
import torchvision

from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls
from .layers import LinearAttention2d


class CTNet(torchvision.models.ResNet):
    def __init__(self, num_classes=2, embeddings=False, pretrained=True):
        super(CTNet, self).__init__(
            block=torchvision.models.resnet.Bottleneck, layers=[3, 4, 6, 3]
        )
        self.num_classes = num_classes
        self.embeddings = embeddings
        self.out_features = 3072

        # Check if use pretrained ResNet50 model (on ImageNet)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
            self.load_state_dict(state_dict)

        # Introduce other bottleneck blocks
        self.layer5 = self._make_layer(
            block=torchvision.models.resnet.Bottleneck,
            planes=512, blocks=3, stride=1
        )

        # Initialize the linear attentions
        self.attention1 = LinearAttention2d(2048, 1024)
        self.attention2 = LinearAttention2d(2048, 2048)

        # Initialize the last bottleneck blocks
        for m in self.layer5.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Re-instantiate the fully connected layer
        del self.fc
        if not self.embeddings:
            self.fc = torch.nn.Linear(self.out_features, self.num_classes)

    def _forward_impl(self, x, attention=False):
        # ResNet50 requires 3-channels input
        x = torch.cat([x, x, x], dim=1)

        # Forward through the input convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Forward through the first two ResNet50 layer
        x = self.layer1(x)
        x = self.layer2(x)

        # Forward through the innermost two ResNet50 layers to get local feature tensors
        l1 = self.layer3(x)
        l2 = self.layer4(l1)

        # Forward through the last layer
        x = self.layer5(l2)

        # Forward through the average pooling to get global feature vectors
        g = self.avgpool(x)

        # Forward through the attention layers
        a1, g1 = self.attention1(l1, g)
        a2, g2 = self.attention2(l2, g)

        # Concatenate the weighted and normalized compatibility scores
        x = torch.cat([g1, g2], dim=1)

        # Pass through the linear classifier, if specified
        if not self.embeddings:
            x = self.fc(x)

        # Return the attention map, optionally
        if attention:
            return x, a1, a2
        return x

    def forward(self, x, attention=False):
        return self._forward_impl(x, attention=attention)
