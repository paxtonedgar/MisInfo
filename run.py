"""
Main file for running the app. Use
python run.py       to print the app menu or
python run.py -h    to print help
"""

import sys
import logging
import argparse
from typing import List

import src.settings
from src.utils.log_utils import setup_logging
from src.tasks.analysis import test_db_conn


def exit_app() -> None:
    """
    Exit app.

    :return: none
    :rtype: None
    """
    logging.getLogger(__name__).info('Exiting')
    sys.exit()


def menu() -> None:
    """
    Print app menu.

    :return: none
    :rtype: None
    """
    print('\nPossible actions:')
    print('=' * len('Possible actions:'))
    for key in sorted(MENU_ACTIONS):
        print('%s: %s' % (key, MENU_ACTIONS[key].__name__))
    print('\nPlease select option(s):')
    # enable selecting multiple actions which will be run in a sequence
    actions = [i.lower() for i in list(sys.stdin.readline().strip())]
    exec_actions(actions)


def exec_actions(actions: List[str]) -> None:
    """
    Execute selected actions.

    :param actions: list of actions (menu options) to be executed
    :type actions: List[str]
    :return: none
    :rtype: None
    """
    if not actions:
        print('\nNo actions selected')
        MENU_ACTIONS['exit']()
    else:
        print('\nSelected the following options: \n%s' % [
            (key, MENU_ACTIONS[key].__name__)
            if key in MENU_ACTIONS
            else (key, 'invalid action')
            for key in actions
        ])
        for action in actions:
            if action in MENU_ACTIONS:
                MENU_ACTIONS[action]()
            else:
                pass
    menu()


MENU_ACTIONS = {
    'db_conn': test_db_conn,
    'exit': exit_app,
}


if __name__ == '__main__':
    setup_logging()

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', metavar='config_file', type=str,
        help='Which config file to use. Default is config.json'
    )
    parser.add_argument(
        'action', choices=list(MENU_ACTIONS.keys()), metavar='action', type=str,
        nargs='+', help=(
            f'Space separated list of actions to be processed. '
            f'Allowed values are: {", ".join(MENU_ACTIONS.keys())}'
        )
    )
    # if no arguments supplied, print help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    cli_args = parser.parse_args()

    # save config file name
    src.settings.init(
        cfg_file=cli_args.config if cli_args.config else 'config.json'
    )

    # setup logger
    logger = logging.getLogger(__name__)
    logger.info('Application started')
    logger.info('Using %s config file', src.settings.CONFIG_FILE)

    # either display menu or directly execute actions
    if cli_args.action:
        # append action to exit app after all actions were executed
        if cli_args.action[-1] != 'exit':
            cli_args.action.append('exit')
        exec_actions(cli_args.action)
    else:
        menu()
