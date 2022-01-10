# built-in
import os
import logging

# installed
import pandas as pd
import seaborn as sns
from matplotlib import pylab

# custom
import src.settings
from src.utils.log_utils import setup_logging, LogLevel
from src.utils.config_loader import ConfigLoader, Config

def setup_jupyter(
        root_dir: str, config_path: str = None,
        logging_level: LogLevel = logging.DEBUG
) -> Config:
    """
    Setup needed for Jupyter.

    :param root_dir: [description]
    :type root_dir: str
    :param config_path: [description], defaults to None
    :type config_path: str, optional
    :param logging_level: [description], defaults to logging.DEBUG
    :type logging_level: LogLevel, optional
    :return: [description]
    :rtype: Config
    """
    src.settings.init()
    cfg = ConfigLoader.load_config(config_path)
    print('Config loaded.')
    setup_logging(
        os.path.join(root_dir, 'logging.json'), logging_level=logging_level
    )
    # other setup
    sns.set()
    palette = sns.color_palette('muted')
    sns.set_palette(palette)
    sns.set(rc={'figure.figsize': (12, 8)})

    pd.options.display.float_format = '{:.4f}'.format
    pd.set_option('max_colwidth', 800)
    pd.set_option('display.max_rows', 200)

    params = {
        'legend.fontsize': 16,
        'figure.figsize': (10, 8),
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    }
    pylab.rcParams.update(params)
    print('Setup done')
    return cfg
