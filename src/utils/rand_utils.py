"""
Utilities for working with random numbers.
"""

import os
import time
import random
from typing import Union

import numpy as np


class RandUtils():
    """
    Utilities for working with random numbers.
    """

    @staticmethod
    def set_random_seed(random_seed: Union[str, int] = None):
        """
        Set all random seeds to 'random_seed'. Sets the following seeds: OS,
        numpy, python random, torch (if installed).

        :param random_seed: random seem to use, defaults to 0
        :type random_seed: Union[str, int], optional
        """
        if random_seed is None:
            random_seed = time.time()
            print('Random seed:', random_seed)
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        np.random.seed(int(random_seed))
        random.seed(int(random_seed))
        try:
            import torch
            torch.manual_seed(int(random_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(random_seed))
        except ImportError:
            pass
