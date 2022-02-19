"""Validate CLI functions."""
import unittest

from konfuzio_sdk.cli import CLI_ERROR, main


def test_cli_error():
    """Test if CLI print is available."""
    assert "Please enter a valid command line option." in CLI_ERROR


def test_help():
    """Test to run CLI."""
    with unittest.mock.patch('sys.argv', ['--help']):
        assert main() == -1


def test_without_input():
    """Test to run CLI."""
    with unittest.mock.patch('sys.argv', [None]):
        assert main() == -1


def test_init_project():
    """Test to run CLI."""
    with unittest.mock.patch('sys.argv', ['init']):
        assert main() == -1
