
"""Konfuzio SDK settings."""

import os

from decouple import config

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

KONFUZIO_PROJECT_ID = config('KONFUZIO_PROJECT_ID', default=-1, cast=int)
KONFUZIO_HOST = config('KONFUZIO_HOST', default="https://app.konfuzio.com")
KONFUZIO_USER = config('KONFUZIO_USER', default=None)
KONFUZIO_TOKEN = config('KONFUZIO_TOKEN', default=None)
KONFUZIO_DATA_FOLDER = config('KONFUZIO_DATA_FOLDER', default='data')

