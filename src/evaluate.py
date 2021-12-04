import json
import os
import yaml
import torch

from tqdm import tqdm
from sklearn import metrics
from covidx.ct.dataset import load_datasets
from covidx.ct.models import CTNet
from covidx.utils.plot import save_attention_map

MODELS_PATH = 'models'
MODEL_NAME = 'ct_net.pt'

if __name__ == '__main__':
    # Load the experiment's parameters
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)
        params = params['evaluation']
    data_path = params['data_path']
    n_attention_maps = int(params['n_attention_maps'])

    # Load the test data set
    _, _, test_data = load_datasets(data_path, num_classes=3)

    # Instantiate the model and load from folder
    model = CTNet(num_classes=3)
    state_filepath = os.path.join(MODELS_PATH, MODEL_NAME)
    model.load_state_dict(torch.load(state_filepath)['model'])

    # Get the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test using device: ' + str(device))

    # Move the model to device
    model.to(device)

    # Make sure the model is set to evaluation mode
    model.eval()

    # Create the metrics path
    metrics_path = 'metrics'
    os.makedirs('metrics', exist_ok=True)

    # Create the attention maps path
    attentions_path = os.path.join(metrics_path, 'attentions')
    os.makedirs(attentions_path, exist_ok=True)

    # Make the predictions for testing the model
    y_pred, y_true = [], []
    with torch.no_grad():
        for idx, (example, label) in enumerate(tqdm(test_data)):
            example = example.unsqueeze(0)
            example = example.to(device)
            pred, att1, att2 = model(example, attention=True)
            pred = torch.log_softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1).item()
            y_pred.append(pred)
            y_true.append(label)
            if idx < n_attention_maps:
                example = (example + 1.0) / 2.0
                filepath = os.path.join(attentions_path, '{}.png'.format(idx))
                save_attention_map(filepath, example, att1, att2)

    # Obtain the classification report
    report = metrics.classification_report(y_true, y_pred, output_dict=True)
    cm = metrics.confusion_matrix(y_true, y_pred)
    metrics = {
        'report': report,
        'confusion_matrix': cm.tolist()
    }

    # Store the metrics in a JSON file
    with open(os.path.join(metrics_path, "ct_net-clf-metrics.json"), 'w') as file:
        json.dump(metrics, file, indent=4)
