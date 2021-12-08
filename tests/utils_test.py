import numpy as np
import io
from PIL import Image


def random_bbox(random_state):
    # Sample bounding boxes randomly
    xmin = random_state.randint(0, 256)
    ymin = random_state.randint(0, 256)
    xmax = random_state.randint(768, 1024)
    ymax = random_state.randint(768, 1024)
    return xmin, ymin, xmax, ymax


def random_image(random_state):
    # Sample an image uniformly
    data = random_state.rand(1024, 1024) * 255
    return data.astype(np.uint8)


def get_formatted_params(random_state):
    xmin, ymin, xmax, ymax = random_bbox(random_state)
    params = {
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax
    }
    params = ['{}={}'.format(k, v) for (k, v) in params.items()]
    return params


def get_image_bytes(random_state):
    image_bytes = io.BytesIO()
    data = random_image(random_state)
    Image.fromarray(data).save(image_bytes, format='png')
    image_bytes.seek(0)
    return image_bytes