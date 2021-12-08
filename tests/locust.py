import numpy as np
from locust import HttpUser, task, tag
from tests.utils_test import get_formatted_params, get_image_bytes


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
        params = get_formatted_params(self.random_state)
        image_bytes = get_image_bytes(self.random_state)
        self.client.post(
            '/predict?' + '&'.join(params),
            files=[('file', ('input-image', image_bytes, 'image/png'))]
        )
