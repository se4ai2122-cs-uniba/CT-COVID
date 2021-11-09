import torch

from covidx.ct.models import CTNet
from covidx.ct.dataset import load_datasets
from covidx.utils.train import train_classifier
import yaml
import os

if __name__ == '__main__':

    # Instantiate the model
    model = CTNet(num_classes=3)
    print(model)
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

    # Load the datasets
    train_data, valid_data, test_data = load_datasets(data_path)

    # Train the classifier
    history = train_classifier(
        model, train_data, valid_data, chkpt_path='ct-checkpoints/ct-resnet50-att2.pt',
        lr=lr, optimizer=optimizer, batch_size=batch_size, epochs=epochs, patience=patience, seed=seed,
        steps_per_epoch=1000, n_workers=2
    )
    # Save the checkpoint
    MODEL_DIR = 'models'

    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    torch.save({
        'model': model.state_dict(),
        'history': history
    }, os.path.join(MODEL_DIR, 'ct_net.pt'))

