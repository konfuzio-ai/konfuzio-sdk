"""Validate cli functions."""

import subprocess
import unittest


class TestCLI(unittest.TestCase):
    """Test CLI functions."""

    def test_cli(self):
        """Test if CLI can be called."""
        proc = subprocess.Popen(
            ['konfuzio_sdk', ''], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        process_result, err = proc.communicate()

        # If stderr is not none, use the original image.
        assert "Please enter a valid command line option." in str(process_result)
