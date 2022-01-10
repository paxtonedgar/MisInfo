"""
Takes care of loading data from project database.
"""

import csv
import logging
from typing import List
from datetime import datetime as dt

import pandas as pd
import elasticsearch


class DataLoader():
    """
    Takes care of loading data from project database.
    """

    _SCROLL_SIZE = 1000
    _KEEP_ALIVE = '5m'

    def __init__(
            self, user: str, password: str, port: int, index_name: str,
            hosts: List[str]
    ) -> None:
        """
        Initialize class.

        :return: none
        :rtype: None
        """
        self._logger = logging.getLogger(__name__)
        self._conn = elasticsearch.Elasticsearch(
            hosts=[{'host': host} for host in hosts],
            http_auth=(user, password),
            port=port,
            timeout=30.0,
        )
        self._index = index_name
        self._logger.info('Created DB connection: %s', self._conn)
        # disable debug messages
        logging.getLogger('elasticsearch').setLevel(logging.INFO)
        logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)

    def _normalize_labels(
            self, data_df: pd.DataFrame, label_map_file: str
    ) -> pd.DataFrame:
        """
        Normalize labels between different fact-checkers.

        :param data_df: [description]
        :type data_df: pd.DataFrame
        :return: [description]
        :rtype: pd.DataFrame
        """
        if 'fact_original' not in data_df:
            return data_df
        fact_map = {}
        with open(label_map_file) as fp_in:
            reader = csv.reader(fp_in, delimiter='\t')
            fact_map = {row[0]: row[1] for row in reader}
        fact_map[''] = 'other'
        fact_map['nan'] = 'other'
        fact_new = []
        for row in data_df.iterrows():
            f_orig = str(row[1]['fact_original']).lower().split('\n')[0]
            if f_orig not in fact_map:
                fact_new.append('other')
            else:
                fact_new.append(fact_map[f_orig])
        data_df['fact_new'] = fact_new
        return data_df

    def load_data(self, label_map_file: str) -> pd.DataFrame:
        """
        Load documents from ElasticSearch.

        :return: [description]
        :rtype: List[Document]
        """
        query = {
            "query": {
                "match_all": {}
            }
        }

        self._logger.info('Loading documents from %s', self._conn)
        data = self._conn.search(
            index=self._index,
            scroll=DataLoader._KEEP_ALIVE,
            size=DataLoader._SCROLL_SIZE,
            body=query
        )

        output = []

        while len(data['hits']['hits']) > 0:
            output.extend(data['hits']['hits'])
            data = self._conn.scroll(
                scroll_id=data['_scroll_id'],
                scroll=DataLoader._KEEP_ALIVE
            )

        # clean up scroll after consuming data
        self._conn.clear_scroll(scroll_id=data['_scroll_id'])

        self._logger.info('Loaded %d documents', len(output))
        self._logger.info('Converting to DataFrame')
        data_df = pd.DataFrame.from_dict(
            {item['_id']: item['_source'] for item in output},
            orient='index'
        )

        # basic data cleanup
        if 'fact' in data_df:
            data_df['fact'] = data_df['fact'].str.lower()
            data_df.loc[:, 'date'] = pd.to_datetime(
                data_df['date'], format='%Y/%m/%d', errors='raise'
            )
        return self._normalize_labels(data_df, label_map_file)

    def _format_dates(self, esoc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all dates in ESOC dataset to uniform format.

        :param esoc_df: [description]
        :type esoc_df: pd.DataFrame
        :return: [description]
        :rtype: pd.DataFrame
        """
        dates = []
        for date in esoc_df['Publication_Date']:
            formatted = None
            try:
                formatted = dt.strptime(date, '%d-%b-%y')
            except ValueError:
                try:
                    formatted = dt.strptime(date, '%d %b %Y')
                except ValueError:
                    try:
                        formatted = dt.strptime(date, '%d/%m/%Y')
                    except ValueError:
                        try:
                            formatted = dt.strptime(date, '%d-%b-%Y')
                        except ValueError:
                            try:
                                formatted = dt.strptime(date, '%d-%b%Y')
                            except ValueError:
                                try:
                                    formatted = dt.strptime(date, '%d-%B-%y')
                                except ValueError:
                                    formatted = None
            dates.append(formatted)
        esoc_df.loc[:, 'Publication_Date'] = dates
        return esoc_df

    def load_esoc_data(self, dataset_path: str) -> pd.DataFrame:
        """
        Load latest version of ESOC dataset.

        :return: [description]
        :rtype: pd.DataFrame
        """
        esoc_data = []
        with open(dataset_path) as fp:
            reader = csv.DictReader(fp)
            esoc_data = {
                row['\ufeff']: dict(row)
                for row in reader if row['Reported_On'] != ''
            }
        esoc_df = pd.DataFrame.from_dict(esoc_data, orient='index')
        return self._format_dates(esoc_df)
