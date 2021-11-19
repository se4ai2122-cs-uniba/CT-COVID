import os
import yaml
import torch
import random
import numpy as np

from covidx.ct.models import CTNet
from covidx.ct.dataset import load_datasets
from covidx.utils.train import train_classifier

if __name__ == '__main__':
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
    print(model)

    # Load the datasets
    train_data, valid_data, test_data = load_datasets(data_path)

    # Train the classifier
    history = train_classifier(
        model, train_data, valid_data, chkpt_path='ct-checkpoints/ct-resnet50-att2.pt',
        lr=lr, optimizer=optimizer, batch_size=batch_size, epochs=epochs, patience=patience,
        steps_per_epoch=1000, n_workers=2
    )

    # Save the best model and training history
    MODEL_DIR = 'models'
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    model = model.cpu()
    torch.save({
        'model': model.state_dict(),
        'history': history
    }, os.path.join(MODEL_DIR, 'ct_net.pt'))
