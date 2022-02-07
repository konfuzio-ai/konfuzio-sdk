"""Central place to collect settings from projects and make them available in the konfuzio_sdk package."""

import os
import sys

# This settings file will search for the settings file in the project in which konfuzio_sdk is imported
sys.path.append(os.getcwd())

KONFUZIO_PROJECT_ID = None
KONFUZIO_USER = None
KONFUZIO_TOKEN = None
KONFUZIO_HOST = "https://app.konfuzio.com"
KONFUZIO_DATA_FOLDER = None
DATA_ROOT = None
FILE_ROOT = None
BASE_DIR = os.getcwd()

try:
    from settings import *  # NOQA
except ImportError:
    pass  # if there is no settings.py in the working directory

if KONFUZIO_DATA_FOLDER:
    DATA_ROOT = os.path.join(BASE_DIR, KONFUZIO_DATA_FOLDER)
    FILE_ROOT = os.path.join(DATA_ROOT, 'pdf')

# Supported File Types in OCR

PDF_FILE = 1
IMAGE_FILE = 2
OFFICE_FILE = 3

SUPPORTED_FILE_TYPES = {PDF_FILE: 'PDF', IMAGE_FILE: 'IMAGE', OFFICE_FILE: 'OFFICE'}
