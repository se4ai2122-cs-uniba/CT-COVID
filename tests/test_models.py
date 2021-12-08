import pytest
import torch
from itertools import product
from covidx.ct.models import CTNet


@pytest.mark.parametrize(
    "num_classes, embeddings, pretrained",
    list(product([2, 3], [False, True], [False, True]))
)
def test_ct_net(num_classes, embeddings, pretrained):
    batch_size, height, width = 8, 224, 224
    model = CTNet(num_classes, embeddings, pretrained)
    inputs = torch.rand(batch_size, 1, height, width)
    outputs = model(inputs)
    if embeddings:
        assert outputs.shape == (batch_size, model.out_features)
    else:
        assert outputs.shape == (batch_size, num_classes)
