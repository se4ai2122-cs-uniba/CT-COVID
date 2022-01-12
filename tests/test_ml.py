import json
import pytest
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report

TEST_LABELS_FILEPATH = "data/ct/test.csv"
CTNET_METRICS_FILEPATH = "metrics/ct_net-clf-metrics.json"


@pytest.mark.ml
def test_metrics_baseline():
    # Load the test set labels file
    true_labels = pd.read_csv(TEST_LABELS_FILEPATH).drop(columns='filename').to_numpy()

    # Predict always class 2 as baseline, i.e. always COVID - 19
    pred_labels = np.full(shape=len(true_labels), fill_value=2)

    # Computes the baseline's weighted averaged F1 score
    baseline_metrics = classification_report(true_labels, pred_labels, zero_division=0, output_dict=True)
    baseline_f1 = baseline_metrics['weighted avg']['f1-score']

    # Load the latest model's evaluation metrics from JSON file
    with open(CTNET_METRICS_FILEPATH, "r") as fp:
        metrics = json.load(fp)

    # Compare the model's F1 vs. baseline's F1, i.e. it achieves at least 5x improvement
    model_f1 = metrics['report']['weighted avg']['f1-score']
    assert model_f1 > 5 * baseline_f1
