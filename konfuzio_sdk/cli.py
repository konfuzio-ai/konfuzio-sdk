"""Command Line interface to the konfuzio_sdk package."""

import argparse
import logging
import sys
from getpass import getpass

from konfuzio_sdk.api import create_new_project, init_env
from konfuzio_sdk.data import Project

sys.tracebacklimit = 0

logger = logging.getLogger(__name__)

CLI_EPILOG = """
These commands should be run inside of your working directory.\n
A bug report can be filed at https://github.com/konfuzio-ai/konfuzio-sdk/issues. Thanks!
"""


def parse_args(parser):
    """Parse command line arguments using sub-parsers for each command."""
    subparsers = parser.add_subparsers(dest='command')

    # Sub-parser for init command
    init_parser = subparsers.add_parser('init', help='Initialize the SDK with user credentials')
    init_parser.add_argument('--user', help='Username for Konfuzio Server', default=None)
    init_parser.add_argument('--password', help='Password for Konfuzio Server', default=None)
    init_parser.add_argument('--host', help='Server Host URL', default=None)

    # Sub-parser for create_project command
    create_project_parser = subparsers.add_parser('create_project', help='Create a new project')
    create_project_parser.add_argument('name', help='Name of the new project')

    # Sub-parser for export_project command
    export_project_parser = subparsers.add_parser('export_project', help='Export project data')
    export_project_parser.add_argument('id', help='ID of the project to export', type=int)
    export_project_parser.add_argument('--include_ai', action='store_true', help='Include AI models in the export')

    return parser.parse_args()


def credentials(args):
    """Retrieve user input or use CLI arguments."""
    user = args.user if args.user else input('Username you use to login to Konfuzio Server: ')
    password = args.password if args.password else getpass('Password you use to login to Konfuzio Server: ')
    host = (
        args.host
        if args.host
        else str(
            input('Server Host URL (press [ENTER] for https://app.konfuzio.com): ') or 'https://app.konfuzio.com'
        ).rstrip('/')
    )
    return user, password, host


def main():
    """CLI of Konfuzio SDK."""
    parser = argparse.ArgumentParser(description='CLI for Konfuzio SDK', epilog=CLI_EPILOG)
    args = parse_args(parser)
    if args.command == 'init':
        user, password, host = credentials(args)
        init_env(user=user, password=password, host=host)
    elif args.command == 'export_project':
        project = Project(id_=args.id)
        project.export_project_data(include_ais=args.include_ai)
    elif args.command == 'create_project':
        create_new_project(args.name)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()  # pragma: no cover
