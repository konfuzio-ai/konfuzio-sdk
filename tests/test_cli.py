"""Validate CLI functions."""
from unittest import mock

import pytest

from konfuzio_sdk.cli import CLI_ERROR, main


def test_cli_error():
    """Test if CLI print is available."""
    assert "Please enter a valid command line option." in CLI_ERROR


def test_help():
    """Test to run CLI."""
    with mock.patch('sys.argv', ['file', '--help']):
        assert main() == -1


def test_without_input():
    """Test to run CLI."""
    with mock.patch('sys.argv', ['file', None]):
        assert main() == -1


def test_init_project():
    """Test to run CLI."""
    with mock.patch('sys.argv', ['file', 'init']):
        with pytest.raises(OSError) as e:
            assert main() == 0
            assert 'reading from stdin while output is captured' in e


def test_export_project():
    """Test to run CLI."""
    with mock.patch('sys.argv', ['file', 'export_project']):
        assert main() == -1


def test_export_project_with_no_int():
    """Test to run CLI."""
    with mock.patch('sys.argv', ['file', 'export_project', 'xx']):
        assert main() == -1


def test_export_project_with_int():
    """Test to run CLI."""
    with mock.patch('sys.argv', ['file', 'export_project', '46']):
        assert main() == 0
