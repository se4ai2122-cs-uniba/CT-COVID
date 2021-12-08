import os
import shutil
import tempfile
from tempfile import NamedTemporaryFile

import torch
from covidx.utils.torch import get_optimizer, RunningAverageMetric, EarlyStopping
import numpy as np
import pytest

from covidx.ct.models import CTNet
from evaluate import MODELS_PATH, MODEL_NAME


@pytest.mark.utils
def test_optimizer():
    assert issubclass(get_optimizer('sgd'), torch.optim.Optimizer)
    with pytest.raises(KeyError):
        get_optimizer('optimizer')


@pytest.mark.utils
def test_RunningAverage():
    test_metric = RunningAverageMetric(1)
    random_list = np.random.rand(5)
    for pos in range(1, len(random_list)):
        mean = np.mean(random_list[:pos])
        test_metric.__call__(random_list[pos-1])
        test_mean = test_metric.average()
        assert test_mean == mean


@pytest.mark.utils
def test_EarlyStopping():

    delta = 1e-4
    patience = 5

    model = CTNet(num_classes=3)
    state_filepath = os.path.join(MODELS_PATH, MODEL_NAME)
    model.load_state_dict(torch.load(state_filepath)['model'])
    path = os.path.join(tempfile.mkdtemp(), 'something.pt')

    with pytest.raises(ValueError) as e:
        EarlyStopping(model, path, -1, delta)
        assert e.message == "The patience value must be positive"
    with pytest.raises(ValueError) as e:
        EarlyStopping(model, path, 1, -1)
        assert e.message == "The delta value must be positive"

    test_patience = EarlyStopping(model, path, patience, delta)
    losses = [1, 1, 2, 1, 1, 1, 1]
    for i in range(patience):
        test_patience.__call__(losses[i])
        assert test_patience.should_stop is False
    test_patience.__call__(losses[patience])
    assert test_patience.should_stop is True
