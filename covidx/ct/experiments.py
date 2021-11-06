from covidx.ct.models import CTNet
from covidx.ct.dataset import load_datasets
from covidx.utils.train import train_classifier
import yaml

if __name__ == '__main__':
    # Load the datasets
    train_data, valid_data, test_data = load_datasets()

    # Instantiate the model
    model = CTNet(num_classes=3)
    print(model)
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)
        params = params['training']

    lr = params['lr']
    seed = params['seed']
    optimizer = params['optimizer']
    batch_size = params['batch_size']
    epochs = params['epochs']
    patience = params['patience']
    # Train the classifier
    train_classifier(
        model, train_data, valid_data, chkpt_path='ct-checkpoints/ct-resnet50-att2.pt',
        lr=lr, optimizer=optimizer, batch_size=batch_size, epochs=epochs, patience=patience, seed=seed,
        steps_per_epoch=1000, n_workers=2
    )
