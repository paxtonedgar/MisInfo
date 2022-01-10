"""
Utilities for working with files (e.g. creation of dated folders)
"""

import os
from datetime import datetime

class FileUtils(object):
    """
    Utilities for working with files (e.g. creation of dated folders)
    """

    @staticmethod
    def ensure_dir(dir_path: str) -> None:
        """
        Create directory (including parent directories) if it does not exist.

        :param dir_path: directory to create
        :type dir_path: str
        :return: None
        :rtype: None
        """
        os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    def mkdir_dated(parent_directory: str, suffix: str = None) -> str:
        """
        Create a new directory inside parent_directory which uses today's
        date its name. Date format: %Y%m%d

        :param parent_directory: where to create the directory
        :type parent_directory: str
        :return: path to the created directory
        :rtype: str
        """
        today = datetime.today()
        dir_name = (
            f"{today.strftime('%Y%m%d')}_{suffix}"
            if suffix else today.strftime('%Y%m%d')
        ) 
        dir_path = os.path.join(parent_directory, dir_name)
        FileUtils.ensure_dir(dir_path)
        return dir_path

    @staticmethod
    def mkdir_timed(
            parent_directory: str, dt: datetime = datetime.now(),
            suffix: str = None
    ) -> str:
        """
        Create a new directory inside parent_directory which uses
        date & time as its name. Date format: %Y%m%d%H%m

        :param parent_directory: where to create the directory
        :type parent_directory: str
        :param dt: datetime.datetime object, default: datetime.now()
        :type dt: datetime.datetime
        :param suffix: optional suffix for the name, default: None
        :type suffix: str, optional
        :return: path to the created directory
        :rtype: str
        """
        dir_name = (
            f"{dt.strftime('%Y%m%d%H%M')}_{suffix}"
            if suffix else dt.strftime('%Y%m%d%H%M')
        )
        dir_path = os.path.join(parent_directory, dir_name)
        FileUtils.ensure_dir(dir_path)
        return dir_path
