import io
import numpy as np
from PIL import Image
from locust import HttpUser, task, tag


class RandomUser(HttpUser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = np.random.RandomState()

    @task(1)
    @tag('welcome')
    def root(self):
        self.client.get('/')

    @task(1)
    @tag('docs')
    def docs(self):
        self.client.get('/docs')

    @task(5)
    @tag('models')
    def models(self):
        self.client.get('/models')

    @task(20)
    @tag('prediction')
    def prediction(self):
        xmin, ymin, xmax, ymax = self.random_bbox()
        params = {
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        }
        params = ['{}={}'.format(k, v) for (k, v) in params.items()]
        image_bytes = io.BytesIO()
        data = self.random_image()
        Image.fromarray(data).save(image_bytes, format='png')
        image_bytes.seek(0)
        self.client.post(
            '/predict?' + '&'.join(params),
            files=[('file', ('input-image', image_bytes, 'image/png'))]
        )

    def random_bbox(self):
        # Sample bounding boxes randomly
        xmin = self.random_state.randint(0, 256)
        ymin = self.random_state.randint(0, 256)
        xmax = self.random_state.randint(768, 1024)
        ymax = self.random_state.randint(768, 1024)
        return xmin, ymin, xmax, ymax

    def random_image(self):
        # Sample an image uniformly
        data = self.random_state.rand(1024, 1024) * 255
        return data.astype(np.uint8)
