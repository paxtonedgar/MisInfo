
import torch

import numpy as np

from src.trainers.base_trainer import BaseTrainer
from src.evaluation.metrics import Metrics


class LSTMTrainer(BaseTrainer):
    """
    Trainer class. Optimizer is by default handled by BaseTrainer.
    """
    def __init__(self, model, config):
        super(LSTMTrainer, self).__init__(model, config)
        self._log_interval = config['log_interval']
        self._batch_size = config['dataloader_params']['batch_size']
        self._logger.info('Batch size: %d', self._batch_size)

    def _train_epoch(self, epoch, train_iter, dev_iter):
        """
        :param epoch:
        :param train_iter:
        :param dev_iter:
        :return:
        """
        # turn on training mode which enables dropout
        self._model.train()

        total_loss = 0
        predicted_values = []
        target_values = []

        labels = np.arange(self._model.num_classes)

        for batch_idx, batch in enumerate(train_iter):
            (data, lengths), target = self._to_tensor(batch.text, batch.label)

            self._optimizer.zero_grad()
            output = self._model(data)
            loss = self._loss_function(output, target, reduction='sum')
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()

            predictions = torch.max(output, 1)[1].view(target.size())
            predicted_values.extend(predictions.data.tolist())
            target_values.extend(target.data.tolist())

            if (batch_idx + 1) % self._log_interval == 0:
                results = Metrics.metrics(
                    predicted_values, target_values, labels
                )
                self._logger.info(
                    'Epoch: {:3d} [{:5d}/{:5.0f} batches] '
                    'Current loss: {:5.6f}, Total average loss: {:5.6f}, '
                    'Accuracy: {:5.2f}'.format(
                        epoch, (batch_idx + 1),
                        len(train_iter.dataset) / self._batch_size,
                        loss.item() / self._batch_size,
                        total_loss / results['n_samples'],
                        results['accuracy']
                    )
                )

        results_train = Metrics.metrics(predicted_values, target_values, labels)
        results_train['loss'] = total_loss / results_train['n_samples']
        results_val, _ = self.evaluate(dev_iter)

        log = {'epoch': epoch}
        log.update({'train_{}'.format(k): v for k, v in results_train.items()})
        log.update({'val_{}'.format(k): v for k, v in results_val.items()})

        return log

    def evaluate(self, data_iter):
        """
        Validate after training an epoch
        :param data_iter:
        :return:
        """
        # switch to evaluation mode (won't dropout)
        self._model.eval()

        total_loss = 0
        predicted_values = []
        target_values = []

        labels = np.arange(self._model.num_classes)

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_iter):
                (data, lengths), target = self._to_tensor(
                    batch.text, batch.label
                )

                output = self._model(data)
                loss = self._loss_function(output, target, reduction='sum')

                total_loss += loss.item()

                predictions = torch.max(output, 1)[1].view(target.size())
                predicted_values.extend(predictions.data.tolist())
                target_values.extend(target.data.tolist())

        results = Metrics.metrics(predicted_values, target_values, labels)
        results['loss'] = total_loss / results['n_samples']

        self._logger.info(
            'Evaluation: Loss: {:5.6f}, Accuracy: {:5.2f}% ({}/{})'.format(
                results['loss'], results['accuracy'],
                results['correct'], results['n_samples']
            )
        )

        return results, predicted_values
