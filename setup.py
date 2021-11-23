"""Setup."""
import sys
import textwrap
from os import path

import setuptools

# check python version
CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 6)

if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write(
        textwrap.dedent(
            f"""
    ==========================
    Unsupported Python version
    ==========================
    This version of Konfuzio SDK requires Python {REQUIRED_PYTHON}, but you're trying to
    install it on Python {CURRENT_PYTHON}.
    """
        )
    )
    sys.exit(1)

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="konfuzio_sdk",
    version="0.1.14",
    author='Helm & Nagel GmbH',
    author_email="info@helm-nagel.com",
    description="Konfuzio Software Development Kit",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/konfuzio-ai/document-ai-python-sdk",
    packages=['konfuzio_sdk'],
    include_package_data=True,
    entry_points={'console_scripts': ['konfuzio_sdk=konfuzio_sdk.cli:main']},
    install_requires=[
        'filetype',
        'nltk',
        'numpy',
        'pandas',
        'Pillow',
        'python-dateutil',
        'python-decouple',
        'requests',
        'tabulate',
        'tqdm',
    ],
    extras_require={
        'dev': [
            'flake8',
            'pydocstyle',
            'pytest',
            'pre-commit',
            'sphinx',
            'sphinx-reload',
            'sphinx-notfound-page',
            'm2r2',
            'sphinx-sitemap',
            'sphinx_rtd_theme',
        ]
    },
)
