"""Setup."""
import subprocess
import sys
import textwrap
from os import path, getenv

import setuptools


# Define version or calculate it for nightly build.
#
# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y.0   # For first release after an increment in Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.Y.ZaN   # Alpha release
#   X.Y.ZbN   # Beta release
#   X.Y.ZrcN  # Release Candidate
#   X.Y.Z     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#

with open(path.join('konfuzio_sdk', 'VERSION')) as version_file:
    version_number = version_file.read().strip()

if getenv('NIGHTLY_BUILD'):
    # create a pre-release
    last_commit = (
        subprocess.check_output(['git', 'log', '-1', '--pretty=%cd', '--date=format:%Y%m%d%H%M%S'])
        .decode('ascii')
        .strip()
    )
    version = f"{version_number}.dev{last_commit}"
else:
    version = f"{version_number}"

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 7)

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
    version=version,
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
        'cloudpickle==2.2.1',  # Used to pickle objects
        'filetype==1.0.7',  # Used to check that files are in the correct format
        'lz4',  # Used to compress pickles
        'nltk',
        'numpy>=1.22.4',
        'pandas>=1.3.5,<2.0.0',
        'Pillow>=8.4.0',
        'python-dateutil',
        'python-decouple',  # todo add ==3.3 ?
        'requests',  # todo add ==2.24.0 ?
        'regex>=2020.6.8',  # re module but better
        'tabulate>=0.9.0',  # Used to pretty print DataFrames
        'tensorflow-cpu',
        'torch>=1.8',
        'torchvision>=0.9',
        'transformers>=4.21.2',  # huggingface transformers and tokenizers
        'tqdm',
        'pympler>=1.0.1',  # Use to get pickle file size.
        'pydantic==1.10.8',  # pydantic is used by spacy. We need to force a higher pydantic version to avoid
        # https://github.com/tiangolo/fastapi/issues/5048
        'scikit-learn>=1.0.2',
        'timm==0.6.7',  # for extra pytorch models, i.e. EfficientNet
        'spacy>=2.3.5',  # used for spaCy tokenization
    ],
    extras_require={
        'dev': [
            'flake8',
            'pydocstyle',
            'pytest',
            'pre-commit',
            'parameterized',
            'Sphinx==4.4.0',
            'sphinx-reload==0.2.0',
            'sphinx-notfound-page==0.8',
            'm2r2==0.3.2',
            'sphinx-sitemap==2.2.0',
            'sphinx-rtd-theme==1.0.0',
            'sphinxcontrib-mermaid==0.8.1',
            'matplotlib==3.7.1',
        ]
    },
)
