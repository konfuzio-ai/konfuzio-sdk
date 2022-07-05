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
    version="0.2.3",
    author='Helm & Nagel GmbH',
    author_email="info@helm-nagel.com",
    description="Konfuzio Software Development Kit",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/konfuzio-ai/konfuzio-sdk/",
    packages=['konfuzio_sdk', 'konfuzio_sdk.tokenizer', 'konfuzio_sdk.trainer'],
    include_package_data=True,
    entry_points={'console_scripts': ['konfuzio_sdk=konfuzio_sdk.cli:main']},
    install_requires=[
        'cloudpickle==2.0.0',
        'filetype==1.0.7',  # Used to check that files are in the correct format
        'dill==0.3.2',  # Used to pickle objects
        'nltk',
        'numpy==1.22.0',
        'pandas',  # todo add ==1.1.5, which causes conflict konfuzio-sdk[dev] 0.2.3 depends on pandas==1.1.5 / 1.0.5
        'Pillow',
        'python-dateutil',
        'python-decouple',  # todo add ==3.3 ?
        'requests',  # todo add ==2.24.0 ?
        'regex==2020.6.8',  # re module but better
        'tabulate==0.8.7',  # Used to pretty print DataFrames
        'tqdm',
        'pathos==0.2.6',
        'pympler==0.9',  # Use to get pickle file size.
        'scikit-learn==0.23.1',
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
