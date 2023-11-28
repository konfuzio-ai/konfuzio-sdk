"""Command Line interface to the konfuzio_sdk package."""

import logging
import sys
from getpass import getpass

from konfuzio_sdk.api import create_new_project
from konfuzio_sdk.data import Project
from konfuzio_sdk.api import init_env

sys.tracebacklimit = 0

logger = logging.getLogger(__name__)

CLI_ERROR = """
Please enter a valid command line option.
----------------------------------------

konfuzio_sdk init
    add the API Token as .env file to connect to the Konfuzio Server, i.e. Host
konfuzio_sdk create_Project >NAME<
    Create a new Project on the Konfuzio Server. Returns the ID of the new Project.
konfuzio_sdk export_project >ID<
    Download the data from a Project by ID to migrate it to another Host.
konfuzio_sdk export_project >ID< include_ai
    Download the data of a Project by ID to migrate it to another Host, including the (status) Done & Activated AI
    models

These commands should be run inside of your working directory.

A bug report can be filed at https://github.com/konfuzio-ai/document-ai-python-sdk/issues. Thanks!
"""


def credentials():
    """Retrieve user input."""
    user = input("Username you use to login to Konfuzio Server: ")
    password = getpass("Password you use to login to Konfuzio Server: ")
    host = str(
        input("Server Host URL (press [ENTER] for https://app.konfuzio.com): ") or "https://app.konfuzio.com"
    ).rstrip('/')
    return user, password, host


def main():
    """CLI of Konfuzio SDK."""
    _cli_file_path = sys.argv.pop(0)  # NOQA
    if len(sys.argv) == 1 and sys.argv[0] == "init":
        user, password, host = credentials()
        init_env(user=user, password=password, host=host)
    elif len(sys.argv) in range(2, 4) and sys.argv[0] == "export_project" and sys.argv[1].isdigit():
        include_ais = False
        if len(sys.argv) == 3:
            include_ais = True if sys.argv[2] == "include_ai" else False
        project = Project(id_=int(sys.argv[1]))
        project.export_project_data(include_ais=include_ais)
    elif len(sys.argv) == 2 and sys.argv[0] == "create_project" and sys.argv[1]:
        create_new_project(sys.argv[1])
    else:
        print(CLI_ERROR)
        return -1
    return 0


if __name__ == "__main__":
    main()  # pragma: no cover
