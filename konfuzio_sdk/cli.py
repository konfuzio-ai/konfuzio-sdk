"""Command Line interface to the konfuzio_sdk package."""

import getpass
import json
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
konfuzio_sdk download_data 123: downloads the data from the project, use the ID of the project, e.g. 123

These commands should be run inside of your working directory.
"""

SETTINGS_FILE_CONTENT = '''
"""Konfuzio SDK settings."""

import os

from decouple import config

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

KONFUZIO_HOST = config('KONFUZIO_HOST', default="https://app.konfuzio.com")
KONFUZIO_USER = config('KONFUZIO_USER', default=None)
KONFUZIO_TOKEN = config('KONFUZIO_TOKEN', default=None)
'''


def init_settings(project_folder):
    """
    Add settings file to the working directory.

    :param project_folder: root folder of the project where to place the settings file
    """
    settings_file = "settings.py"

    with open(os.path.join(project_folder, settings_file), "w", encoding="utf-8") as f:
        f.write(SETTINGS_FILE_CONTENT)

    return True


def init_env(project_folder):
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


def init(project_folder="./"):
    """
    Add settings and .env files to the working directory.

    :param project_folder: Root folder of the project
    """
    assert init_settings(project_folder)
    assert init_env(project_folder)


def data(id: int):
    """
    Download the data from the project.

    It has to run after having the .env and settings.py files.

    :param include_extra: if to download also the pdf, bounding boxes information and page images of the documents
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


def create_project(token=None, host=None):
    """Create a new project."""
    project_name = input("Name of the project: ")
    response = create_new_project(project_name, token, host)
    if response.status_code == 201:
        project_id = json.loads(response.text)["id"]
        print(
            f"Project {project_name} (ID {project_id}) was created successfully!"
            f"Initializing the environment with the project that was created."
        )
        return project_id
    else:
        raise Exception(f'The project {project_name} was not created.')


def verify_data_folder(project_folder, data_folder):
    """
    Verify if data folder is empty.

    If not empty, asks the user for a new folder name or if to use the same.

    :param project_folder: Root folder of the project
    :param data_folder: Name of the data folder
    :return: Final name for the data folder
    """
    project_data_folder = f"{project_folder}/{data_folder}"

    if os.path.exists(project_data_folder) and len(os.listdir(project_data_folder)) != 0:
        print(
            f"The directory: {project_data_folder} is not empty."
            f"If you choose to continue, the old data can be overwritten."
        )
        update_data_dir = input("Choose another folder to allocate the data? (Y[default]/ N) ")
        # for precaution, everything different than "N" is considered "Y"
        if update_data_dir != 'N':
            data_folder = input("Folder where to allocate the data: ")
            data_folder = verify_data_folder(project_folder, data_folder)

    return data_folder


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
        init()

    elif first_arg == 'download_data':
        try:
            id = cli_options[0]
        except IndexError:
            raise IndexError("Please define a project ID, e.g. 'konfuzio_sdk download_data 123'")
        data(id=int(id))

    else:
        print(CLI_ERROR)


if __name__ == "__main__":
    main()
