
import os
import json
import logging

from src.utils.config_loader import ConfigLoader
from src.data.data_loader import DataLoader

def test_db_conn() -> None:
    cfg = ConfigLoader.load_config()

    output = DataLoader(
        user=cfg['index']['user'],
        password=cfg['index']['password'],
        port=cfg['index']['port'],
        index_name=cfg['index']['index_name'],
        hosts=cfg['index']['hosts']
    ).load_data()

    logger = logging.getLogger(test_db_conn.__name__)
    logger.info('Retrieved %d documents from ES', len(output))

    output_path = os.path.join(cfg['paths']['data_dir'], 'raw/data.json')
    logger.info('Saving output in %s', output_path)
    with open(output_path, 'w') as fp:
        json.dump(output, fp, indent=4)

    logger.info('Example document:\n%s', json.dumps(output[0], indent=4))
