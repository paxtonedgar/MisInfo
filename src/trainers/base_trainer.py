"""
Base trainer class. Takes care of:
* Training
* Checkpoints and resuming
* Logging during training
"""

import os
import math
import json
import logging
from typing import Dict, Any, Optional, Iterable, Type, TypeVar

import torch

import src.settings
from src.models.base_model import BaseModel


BaseTrainerType = TypeVar('BaseTrainerType', bound='BaseTrainer')


class BaseTrainer():
    """
    Base class for all trainers.
    """

    def __init__(self, model: Type[BaseModel], config: Dict[str, Any]) -> None:
        """
        Initialize the trainer. Takes care of:
        * Setting device to CPU or GPU
        * Instantiating optimizer and loss function

        :param config: dictionary
        :type config: Dict[str, Any]
        :return: none
        :rtype: None
        """
        self._logger = logging.getLogger(__name__)

        self._model = model
        self._config = config
        self._logger.info(
            'Training model %s (using %s config)',
            self._model.__class__.__name__, self._config['name']
        )

        # check if training on GPU
        self._cuda = self._config['cuda'] and torch.cuda.is_available()
        if self._config['cuda'] and not torch.cuda.is_available():
            self._logger.warning('CUDA not available, training on CPU.')
        if not self._config['cuda'] and torch.cuda.is_available():
            self._logger.warning('CUDA available but not in use.')

        # select device
        dev_idx = (
            src.settings.DEVICE_ID
            if src.settings.DEVICE_ID is not None
            else self._config['device_idx']
        )
        self._device = torch.device(f'cuda:{dev_idx}' if self._cuda else 'cpu')
        self._model = self._model.to(self._device)
        if self._cuda:
            torch.cuda.set_device(dev_idx)
        self._logger.info(
            'Current device: %s (GPU: %s)',
            self._device, 'Yes' if self._cuda else 'No'
        )

        # setup optimizer
        self._optimizer = getattr(torch.optim, config['optimizer_type'])(
            model.parameters(), **config['optimizer_params']
        )
        self._logger.info('Using optimizer: %s', self._optimizer)

        # setup loss function
        self._loss_function = getattr(
            torch.nn.functional, config['loss_function']
        )
        self._logger.info(
            'Using loss function: %s', self._loss_function.__name__
        )

        # how long to train
        self._start_epoch = 1
        self._checkpoint_epoch = None
        self._epochs = self._config['epochs']
        self._logger.info('Will train %s epochs', self._epochs)

        # saving config
        self._save_interval = self._config['save_interval']
        self._save_directory = self._config['save_directory']
        os.makedirs(self._save_directory, exist_ok=True)
        self._save_config(self._save_directory, self._config)
        self._logger.info(
            'Save interval: %s, save directory: %s',
            self._save_interval, self._save_directory
        )

        # how to identify best model
        self._save_best = self._config['save_best']
        self._monitor_metric = self._config['monitor_metric']
        self._monitor_mode = self._config['monitor_mode']
        assert self._monitor_mode == 'min' or self._monitor_mode == 'max'
        self._monitor_best = (
            math.inf if self._monitor_mode == 'min' else -math.inf
        )

        # full training log
        self._training_log = []

    def _to_tensor(self, data, target):
        """
        :param data:
        :param target:
        :return:
        """
        if self._cuda:
            data = data[0].to(self._device), data[1]
            target = target.to(self._device)
        return data, target

    @property
    def checkpoint_epoch(self) -> Optional[int]:
        """
        If resumed from a checkpoint, epoch in which the checkpoint was created.

        :return: if resumed from a checkpoint, epoch in which the checkpoint
                 was created
        :rtype: Optional[int]
        """
        return self._checkpoint_epoch

    def best_model_path(self) -> str:
        """
        Where to save best model.

        :return: path to best model
        :rtype: str
        """
        return os.path.join(self._save_directory, 'model_best.pt')

    def checkpoint_path(self, epoch: int) -> str:
        """
        Where to save checkpoint.

        :param epoch: checkpoint epoch
        :type epoch: int
        :return: path to checkpoint save
        :rtype: str
        """
        return os.path.join(
            self._save_directory, 'checkpoint-epoch{:03d}.pt'.format(epoch)
        )

    def _save_config(self, save_directory: str, config: Dict[str, Any]) -> None:
        """
        Save model config.

        :param save_directory: where to save the config
        :type: save_directory: str
        :param config: model config
        :type config: Dict[str, Any]
        :return: none
        :rtype: None
        """
        save_path = os.path.join(save_directory, 'model_config.json')
        with open(save_path, 'w') as fp_out:
            json.dump(config, fp_out, indent=4, sort_keys=False)

    def _add_to_log(self, entry: Any) -> None:
        """
        Add entry to training log.

        :param entry: entry to add
        :type entry: Any
        :return: none
        :rtype: None
        """
        self._training_log.append(entry)

    def _resume(
            self, model_state, optim_state, start_epoch: int, monitor_best, 
            log: Dict[int, Any]
    ) -> None:
        """
        :param model_state:
        :param optim_state:
        :param start_epoch:
        :param monitor_best:
        :param log:
        :return:
        """
        self._model.load_state_dict(model_state)
        self._optimizer.load_state_dict(optim_state)
        if self._cuda:
            for state in self._optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self._device)
        self._start_epoch = start_epoch
        self._checkpoint_epoch = start_epoch
        self._monitor_best = monitor_best
        self._training_log = log

    def train(self, train_iter: Iterable, dev_iter: Iterable) -> None:
        """
        Full training logic.

        :param train_iter: iterator over training data
        :type train_iter: Iterable
        :param dev_iter: iterator over validation data
        :type dev_iter: Iterable
        :return: none
        :rtype: None
        """
        for epoch in range(self._start_epoch, self._epochs + 1):
            log = self._train_epoch(epoch, train_iter, dev_iter)
            self._add_to_log(log)
            self._save_log()

            # save best model
            if self._save_best:
                mode = self._monitor_mode
                curr = log[self._monitor_metric]
                best = self._monitor_best
                if (
                        mode == 'min' and curr < best 
                        or mode == 'max' and curr > best
                ):
                    self._monitor_best = curr
                    self._save_checkpoint(epoch, save_best=True)

            # save checkpoint
            if self._save_interval and epoch % self._save_interval == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(
            self, epoch: int, train_iter: Iterable, dev_iter: Iterable
    ):
        """
        Training logic for an epoch

        :param epoch: current epoch number
        :type epoch: int
        :param train_iter: iterator over training data
        :type train_iter: Iterable
        :param dev_iter: iterator over validation data
        :type dev_iter: Iterable
        :raises NotImplementedError: method needs to be implemented by every
                                     subclass
        """
        raise NotImplementedError

    def evaluate(self, data_iter: Iterable):
        """
        Evaluate the model.

        :param data_iter: iterator over data
        :type data_iter: Iterable
        """
        raise NotImplementedError

    def _save_log(self) -> None:
        """
        Save training log.

        :return: none
        :rtype: None
        """
        log_path = os.path.join(self._save_directory, 'log.log')
        with open(log_path, 'w') as fp_out:
            json.dump(self._training_log, fp_out, indent=4, sort_keys=False)

    def _save_checkpoint(self, epoch: int, save_best: bool = False) -> None:
        """
        Saving checkpoints.

        :param epoch: current epoch number
        :type epoch: int
        :param save_best: if True, rename the saved checkpoint to
                          'model_best.pth.tar'
        :type save_best: bool
        :return: none
        :rtype: None
        """
        arch = self._model.__class__.__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'training_log': self._training_log,
            'model_state': self._model.state_dict(),
            'optimizer_state': self._optimizer.state_dict(),
            'monitor_best': self._monitor_best,
            'config': self._config
        }
        file_path = (
            self.best_model_path() if save_best
            else self.checkpoint_path(epoch)
        )
        torch.save(state, file_path)
        self._logger.info(
            'Saving %s: %s',
            'checkpoint' if not save_best else 'current best', file_path
        )

    @classmethod
    def initialize(
            cls: Type[BaseTrainerType], model: Type[BaseModel],
            config: Dict[str, Any], checkpoint_path: str = None
    ) -> Type[BaseTrainerType]:
        """
        Initialize BaseTrainer from a checkpoint.

        :param model: training model
        :type model: BaseModel
        :param config: model config
        :type config: Dict[str, Any]
        :param checkpoint_path: where the checkpoint is stored, defaults to None
        :type checkpoint_path: str
        :return: instance of BaseTrainer
        :rtype: Type[BaseTrainerType]
        """
        logger = logging.getLogger(__name__)

        # if checkpoint path not provided initialize normally and exit
        if not checkpoint_path:
            logger.info('No checkpoint path found, initializing normally')
            return cls(model, config)

        # if checkpoint path provided load from checkpoint
        logger.info('Loading checkpoint from %s', checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cuda' if torch.cuda.is_available() else 'cpu'
        )

        logger.info('Overriding config with config from checkpoint')
        config = checkpoint['config']

        logger.info('Resuming training from saved state')
        trainer = cls(model, config)
        trainer._resume(
            model_state=checkpoint['model_state'],
            optim_state=checkpoint['optimizer_state'],
            start_epoch=checkpoint['epoch'] + 1,
            monitor_best=checkpoint['monitor_best'],
            log=checkpoint['training_log']
        )
        logger.info(
            'Checkpoint %s (epoch %s) loaded',
            checkpoint_path, trainer._start_epoch - 1
        )
        return trainer
