"""Relative import of all settings."""
from .settings_importer import *  # NOQA

import sys
import logging

logger = logging.getLogger(__name__)
if '3.7' in sys.version:
    logger.warning("Some Konfuzio SDK functionalities may not work with Python 3.7. We recommend using Python 3.8.")
