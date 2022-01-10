"""
Class for managing paths.
"""

from os import listdir
from os.path import join
from typing import Dict, Any


class PathManager(object):
    """
    Class for managing paths.
    """

    def __init__(self, config: Dict[str, Any], root: str = None) -> None:
        self._cfg = config['paths']
        self._data_dir = self._cfg['data_dir']
        self._output_dir = self._cfg['output_dir']
        if root is not None:
            self._data_dir = join(root, self._data_dir)
            self._output_dir = join(root, self._output_dir)
        # directories
        self._processed_dir = join(self._data_dir, 'processed/')
        self._raw_dir = join(self._data_dir, 'raw/')

    @property
    def processed_data_dir(self) -> str:
        return self._processed_dir

    def processed_file_path(self, file_name: str) -> str:
        return join(self.processed_data_dir, file_name)

    @property
    def raw_data_dir(self) -> str:
        return self._raw_dir

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @property
    def figures_dir(self) -> str:
        return join(self._output_dir, 'figures')

    def figure_path(self, file_name: str) -> str:
        return join(self.figures_dir, file_name)

    @property
    def esoc_covid_dataset_path(self) -> str:
        esoc_dir = join(self._raw_dir, 'esoc_covid/')
        latest = None
        for fname in listdir(esoc_dir):
            if not fname.endswith('.csv'):
                continue
            if latest is None or latest < fname:
                latest = fname
        return join(esoc_dir, latest)

    @property
    def label_map_file(self) -> str:
        return join(self._raw_dir, 'label_map.tsv')
