"""Central place to collect settings from projects and make them available in the konfuzio_sdk package."""

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
