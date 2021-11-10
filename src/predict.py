import json
import os
import torch

from sklearn import metrics
from covidx.ct.dataset import load_datasets
from covidx.ct.models import CTNet
from covidx.utils.plot import save_attention_map
import yaml

MODELS_PATH = 'models'
MODEL_NAME = 'ct_net.pt'

if __name__ == '__main__':
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)
        params = params['training']

    data_path = params['data_path']
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

    # Make the prediction
    if not os.path.isdir('visualization/ct-attentions'):
        os.mkdir('visualization/ct-attentions')

    y_pred = []
    y_true = []
    with torch.no_grad():
        for idx, (example, label) in enumerate(test_data):
            example = example.unsqueeze(0)
            example = example.to(device)
            pred, map1, map2 = model(example, attention=True)
            pred = torch.log_softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1).item()
            y_pred.append(pred)
            y_true.append(label)
            if idx < 250:
                example = (example + 1) / 2
                filepath = os.path.join('visualization/ct-attentions', '{}.png'.format(idx))
                save_attention_map(filepath, example, map1, map2)

    report = metrics.classification_report(y_true, y_pred, output_dict=True)
    cm = metrics.confusion_matrix(y_true, y_pred)
    metrics = {'report': report,
               'confusion_matrix': cm.tolist()}

    with open(os.path.join(MODELS_PATH, MODEL_NAME) + '-report.json', 'w') as file:
        json.dump(metrics, file, indent=4)

    with open("metrics.json", 'w') as file:
        json.dump(report, file, indent=4)
