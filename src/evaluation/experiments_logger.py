"""
Log experiments.
"""

import json
import logging
import platform
from os.path import join
from typing import Dict, Any, List
from datetime import datetime
import pkg_resources

import numpy as np

from src.utils.file_utils import FileUtils


class ExperimentsLogger(object):
    """
    Log experiments.
    """

    def __init__(self, output_dir: str) -> None:
        """Initialize class.

        :param output_dir: directory for storing results
        :type output_dir: str
        :return: none
        :rtype: None
        """
        self._logger = logging.getLogger(__name__)
        self._output_dir = output_dir

    @staticmethod
    def npconverter(obj):
        """
        [summary]

        :param obj: [description]
        :type obj: [type]
        :return: [description]
        :rtype: [type]
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()

    def get_installed_libraries(self) -> List[str]:
        """Get a list of installed libraries with versions.

        :return: list of libraries and versions
        :rtype: List[str]
        """
        return sorted([
            f'{d.project_name}=={d.version}'
            for d in iter(pkg_resources.working_set)
        ])

    def get_python_version(self) -> str:
        """Return Python version as string.

        :return: python version
        :rtype: str
        """
        return platform.python_version()

    def log_experiment(
            self, app_cfg: Dict[str, Any], model_cfg: Dict[str, Any],
            results: Dict[str, Any], dt: datetime = datetime.now()
    ) -> None:
        """[summary]
        :param results: [description]
        :type results: Dict[str, Any]
        """
        experiment_dir = join(self._output_dir, 'experiments/')
        experiment_dir = FileUtils.mkdir_timed(
            experiment_dir, dt=dt, suffix=model_cfg['name']
        )
        with open(join(experiment_dir, 'app_config.json'), 'w') as fp_in:
            json.dump(app_cfg, fp_in, indent=4)
        with open(join(experiment_dir, 'model_config.json'), 'w') as fp_in:
            json.dump(model_cfg, fp_in, indent=4)
        with open(join(experiment_dir, 'results.json'), 'w') as fp_in:
            json.dump(
                results, fp_in, indent=4,
                default=ExperimentsLogger.npconverter
            )
        with open(join(experiment_dir, 'requirements.txt'), 'w') as fp_in:
            fp_in.write('\n'.join(self.get_installed_libraries()))
        with open(join(experiment_dir, 'system.json'), 'w') as fp_in:
            json.dump({'python': self.get_python_version()}, fp_in, indent=4)
