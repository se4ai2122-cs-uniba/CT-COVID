import os
import tempfile

import torch
from tempfile import NamedTemporaryFile
from PIL import Image
from covidx.utils.plot import save_attention_map


def test_save_attention_map():
    padding = 4
    height, width = 224, 224
    img = torch.rand(1, 1, height, width)
    att1 = torch.rand(1, 1, height // 16, width // 16)
    att2 = torch.rand(1, 1, height // 32, width // 32)
    path = os.path.join(tempfile.mkdtemp(), 'something.png')
    save_attention_map(path, img, att1, att2)
    with Image.open(path) as img:
        assert len(img.getbands()), 3
        assert img.size == (width * 3 + padding * 4, height + padding * 2)
