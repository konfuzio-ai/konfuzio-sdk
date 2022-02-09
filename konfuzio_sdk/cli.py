"""Command Line interface to the konfuzio_sdk package."""

import getpass
import logging
import os
import sys

from tqdm import tqdm

import konfuzio_sdk
from konfuzio_sdk.api import get_auth_token, create_new_project

sys.tracebacklimit = 0

logger = logging.getLogger(__name__)

CLI_ERROR = """
Please enter a valid command line option.
----------------------------------------
Valid options:

konfuzio_sdk init: inits the konfuzio Package by setting the necessary files
konfuzio_sdk create_project my_project: creates a new project online
konfuzio_sdk migrate_data 123: downloads the data from the project, use the ID of the project, e.g. 123

These commands should be run inside of your working directory.
"""


def init_env(project_folder="./"):
    """
    Add the .env file to the working directory.

    :param project_folder: Root folder of the project where to place the .env file
    :return: file content
    """
    user = input("Username you use to login to Konfuzio Server: ")
    password = getpass.getpass("Password you use to login to Konfuzio Server: ")
    host = input("Host from where to get the data (press [ENTER] for the default: https://app.konfuzio.com): ")

    if host == "":
        host = konfuzio_sdk.KONFUZIO_HOST

    token = get_auth_token(user, password, host)

    with open(os.path.join(project_folder, ".env"), "w") as f:
        f.write(f"KONFUZIO_HOST = {host}\n")
        f.write(f"KONFUZIO_USER = {user}\n")
        f.write(f"KONFUZIO_TOKEN = {token}\n")

    print("[SUCCESS] SDK initialized!")

    return True


def data(id: int):
    """
    Migrate your project to another host.

    See https://help.konfuzio.com/integrations/migration-between-konfuzio-server-instances/index.html
        #migrate-projects-between-konfuzio-server-instances
    """
    print("Starting the download. Please wait until the data download is finished...")
    from konfuzio_sdk.data import Project

    training_prj = Project(id=id)
    training_prj.update()

    if len(training_prj.documents + training_prj.test_documents) == 0:
        raise ValueError("No documents in the training or test set. Please add them.")

    for document in tqdm(training_prj.documents + training_prj.test_documents):
        document.get_file()
        document.get_file(ocr_version=False)
        document.get_bbox()
        document.get_images()

    print("[SUCCESS] Data downloading finished successfully!")


def main():
    """CLI of Konfuzio SDK."""
    cli_options = sys.argv
    _cli_file_path = cli_options.pop(0)  # NOQA

    try:
        first_arg = cli_options.pop(0)
    except IndexError:
        print(CLI_ERROR)
        return -1

    if first_arg == 'init':
        init_env()

    elif first_arg == 'migrate_data':
        try:
            id = cli_options[0]
        except IndexError:
            raise IndexError("Please define a project ID, e.g. 'konfuzio_sdk download_data 123'")
        data(id=int(id))

    elif first_arg == 'create_project':
        try:
            project_name = cli_options[0]
        except IndexError:
            raise IndexError("Please define a project name, e.g. 'konfuzio_sdk create_project my_project'")
        create_new_project(project_name)

    else:
        print(CLI_ERROR)


if __name__ == "__main__":
    main()
