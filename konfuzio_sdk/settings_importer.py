"""Central place to collect settings from Projects and make them available in the konfuzio_sdk package."""
import importlib
import logging
import os
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

dependencies = ['cloudpickle', 'transformers', 'torch', 'spacy', 'sklearn', 'torchvision', 'timm', 'tensorflow']
installed_dependencies = []

for dependency in dependencies:
    try:
        successful_import = importlib.import_module(dependency)
        installed_dependencies.append(dependency)
        del successful_import
    except ImportError:
        pass

if 'sklearn' in installed_dependencies:
    EXTRAS_INSTALLED = 'extraction'
elif 'timm' in installed_dependencies:
    EXTRAS_INSTALLED = 'categorization'
elif 'tensorflow' in installed_dependencies:
    EXTRAS_INSTALLED = 'file_splitting'
elif 'transformers' in installed_dependencies and 'sklearn' in installed_dependencies:
    EXTRAS_INSTALLED = 'all'
else:
    EXTRAS_INSTALLED = None


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
