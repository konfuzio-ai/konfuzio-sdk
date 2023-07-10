"""Central place to collect settings from Projects and make them available in the konfuzio_sdk package."""
import logging
import os
import pkg_resources
import sys

from decouple import AutoConfig

sys.path.append(os.getcwd())
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

config = AutoConfig(search_path=os.getcwd())
KONFUZIO_HOST = config('KONFUZIO_HOST', default="https://app.konfuzio.com")
KONFUZIO_USER = config('KONFUZIO_USER', default=None)
KONFUZIO_TOKEN = config('KONFUZIO_TOKEN', default=None)

PDF_FILE = 1
IMAGE_FILE = 2
OFFICE_FILE = 3

SUPPORTED_FILE_TYPES = {PDF_FILE: 'PDF', IMAGE_FILE: 'IMAGE', OFFICE_FILE: 'OFFICE'}

LOG_FILE_PATH = os.path.join(os.getcwd(), 'konfuzio_sdk.log')
LOG_FORMAT = (
    "%(asctime)s [%(name)-20.20s] [%(threadName)-10.10s] [%(levelname)-8.8s] "
    "[%(funcName)-20.20s][%(lineno)-4.4d] %(message)-10s"
)


def is_dependency_installed(dependency: str) -> bool:
    """Check if a package is installed."""
    return dependency in {pkg.key for pkg in pkg_resources.working_set}


OPTIONAL_IMPORT_ERROR = (
    "A library *modulename* has not been found, so Konfuzio SDK is initialized without the AI "
    "components. To install Konfuzio SDK with all the AI-related libraries, see "
    "https://dev.konfuzio.com/sdk/get_started/index.html#install-konfuzio-sdk-package."
)

extras = [
    'chardet==5.1.0',
    'pydantic==1.10.8',
    'torch>=1.8',
    'torchvision>=0.9',
    'transformers>=4.21.2',
    'tensorflow-cpu==2.12.0',
    'timm==0.6.7',
    'spacy>=2.3.5',
]

for extra in extras:
    extra = extra.split()[0].split('>=')[0].split('==')[0]
    is_installed = is_dependency_installed(extra)
    if not is_installed:
        logging.error(OPTIONAL_IMPORT_ERROR.replace('*modulename*', extra))

DO_NOT_LOG_IMPORT_ERRORS = True


def get_handlers():
    """Get logging handlers based on environment variables."""
    handlers = [logging.StreamHandler()]

    if config('LOG_TO_FILE', cast=bool, default=True):
        with open(LOG_FILE_PATH, "a") as f:
            if f.writable():
                handlers.append(logging.FileHandler(LOG_FILE_PATH))

    return handlers


logging.basicConfig(
    level=config('LOGGING_LEVEL', default=logging.INFO, cast=int), format=LOG_FORMAT, handlers=get_handlers()
)
