"""
Utilities for working with config files.
"""

import json
from typing import Dict, Any, NewType
from os.path import dirname, abspath, join, isabs

import src.settings


Config = NewType('Config', Dict[str, Dict[str, Any]])


class ConfigLoader():
    """
    Container for utilities for loading config files.
    """

    _project_root = dirname(dirname(dirname(abspath(__file__))))

    @staticmethod
    def load_config(config_path: str = None) -> Config:
        """
        Load app config.

        :param config_path: path to app config json file, if no path supplied,
                            loads config from 'config.json', defaults to None
        :type config_path: str, optional
        :return: dictionary with app config
        :rtype: Config
        """
        if not config_path:
            # use default config
            config_path = join(
                ConfigLoader._project_root, src.settings.CONFIG_FILE
            )
        if not isabs(config_path):
            config_path = join(ConfigLoader._project_root, config_path)
        with open(config_path) as fp_in:
            cfg = json.load(fp_in)
        return cfg
