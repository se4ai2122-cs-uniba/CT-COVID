import pytest
import numpy as np
import torch
import time
import yaml
import random
from covidx.ct.models import CTNet
from covidx.ct.dataset import load_datasets
from covidx.utils.train import train_classifier
import torch.utils.data as data_utils

@pytest.mark.gpu
def test_train_overfit():
    # Load the experiment's parameters
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)
        params = params['training']
    lr = float(params['lr'])
    seed = int(params['seed'])
    optimizer = params['optimizer']
    batch_size = int(params['batch_size'])
    epochs = int(params['epochs'])
    patience = int(params['patience'])
    data_path = params['data_path']

    # Set all the seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Instantiate the model
    model = CTNet(num_classes=3)

    # Load the datasets
    train_data, valid_data, _ = load_datasets(data_path, augment=False)
    indices = torch.arange(batch_size)
    reduced_train = data_utils.Subset(train_data, indices)
    reduced_valid = data_utils.Subset(valid_data, [0])

    reduced_train.get_targets = lambda: np.arange(model.num_classes)
    reduced_valid.get_targets = lambda: np.arange(model.num_classes)

    history = train_classifier(
        model, reduced_train, reduced_valid, chkpt_path='ct-checkpoints/ct-resnet50-att2.pt',
        lr=lr, optimizer=optimizer, batch_size=batch_size, epochs=epochs, patience=patience,
        steps_per_epoch=1000, n_workers=0, verbose=False
    )
    assert history['train']['accuracy'][-1] == pytest.approx(1, abs=0.05)
