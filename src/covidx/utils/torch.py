import torch
import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given number of consecutive epochs."""
    def __init__(self, model, chkpt_path, patience=1, delta=1e-4):
        """
        Instantiate an EarlyStopping object.

        :param model: The model.
        :param chkpt_path: The filepath of the checkpoint file.
        :param patience: The number of consecutive epochs to wait.
        :param delta: The minimum change of the monitored quantity.
        """
        if patience <= 0:
            raise ValueError("The patience value must be positive")
        if delta <= 0.0:
            raise ValueError("The delta value must be positive")
        self.model = model
        self.chkpt_path = chkpt_path
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0

    @property
    def should_stop(self):
        """
        Check if the training process should stop.
        """
        return self.counter >= self.patience

    def get_best_state(self):
        """
        Get the best model's state dictionary.
        """
        with open(self.chkpt_path, 'rb') as f:
            best_state = torch.load(f)
        return best_state

    def __call__(self, loss):
        """
        Call the object.

        :param loss: The validation loss measured.
        """
        # Check if an improved of the loss happened
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0

            # Save the best model state parameters
            with open(self.chkpt_path, 'wb') as f:
                torch.save(self.model.state_dict(), f)
        else:
            self.counter += 1


class RunningAverageMetric:
    """Running (batched) average metric."""
    def __init__(self, batch_size):
        """
        Initialize a running average metric object.

        :param batch_size: The batch size.
        """
        self.batch_size = batch_size
        self.metric_accumulator = 0.0
        self.n_metrics = 0

    def __call__(self, x):
        """
        Accumulate a metric.

        :param x: The metric value.
        """
        self.metric_accumulator += x
        self.n_metrics += 1

    def average(self):
        """
        Get the metric average.

        :return: The metric average.
        """
        return self.metric_accumulator / (self.n_metrics * self.batch_size)


def get_optimizer(optimizer):
    return {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'rmsprop': torch.optim.RMSprop
    }[optimizer]
