"""Run doctests on modules."""
import os

doctest_file_paths = ['konfuzio_sdk/trainer/file_splitting.py']

for path in doctest_file_paths:
    os.system(f'python -m doctest -v {path}')
