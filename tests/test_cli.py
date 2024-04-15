# ruff: noqa: A002

"""Validate CLI functions."""

import sys
from unittest import TestCase, mock
from unittest.mock import patch
from io import StringIO
import argparse

from konfuzio_sdk.cli import credentials, main, parse_args


class TestCLI(TestCase):
    """Test the konfuzio_sdk CLI."""

    @patch('sys.stdout', new_callable=StringIO)
    def test_help(self, mock_stdout):
        """Test to run CLI."""
        with self.assertRaises(SystemExit) as _:
            # help exits the program, so capture SystemExit exception to inspect the exit code
            with mock.patch('sys.argv', ['file', '--help']):
                main()
        self.assertIn('usage:', mock_stdout.getvalue())
        self.assertIn('--help', mock_stdout.getvalue())

    @patch('sys.stdout', new_callable=StringIO)
    def test_without_input(self, mock_stdout):
        """Test to run CLI."""
        with mock.patch('sys.argv', ['file']):
            main()
        self.assertIn('usage:', mock_stdout.getvalue())
        self.assertIn('--help', mock_stdout.getvalue())

    @patch('builtins.input', side_effect=['myuser', 'https://konfuzio.yourcompany.com'])
    @patch('konfuzio_sdk.cli.getpass', side_effect=['pw'])
    @patch('sys.argv', ['file', 'init'])
    def test_username_password_custom_host(self, input, getpass):
        """Test to init with custom host and without cli args. See https://stackoverflow.com/a/55580216 and 56401696 for patches."""
        parser = argparse.ArgumentParser()
        args = parse_args(parser)
        assert credentials(args) == ('myuser', 'pw', 'https://konfuzio.yourcompany.com')

    @patch('builtins.input', side_effect=['myuser', ''])
    @patch('konfuzio_sdk.cli.getpass', side_effect=['pw'])
    @patch('sys.argv', ['file', 'init'])
    def test_username_password_default_host(self, input, getpass):
        """Test to init with default host and without cli args."""
        parser = argparse.ArgumentParser()
        args = parse_args(parser)
        assert credentials(args) == ('myuser', 'pw', 'https://app.konfuzio.com')

    @patch('builtins.input', side_effect=['myuser', 'https://konfuzio.yourcompany.com//'])
    @patch('konfuzio_sdk.cli.getpass', side_effect=['pw'])
    @patch('sys.argv', ['file', 'init'])
    def test_removing_trailing_slashes(self, input, getpass):
        """Test init to ensure that any trailing slash(es) in the end of the host URL are removed."""
        parser = argparse.ArgumentParser()
        args = parse_args(parser)
        assert credentials(args) == ('myuser', 'pw', 'https://konfuzio.yourcompany.com')

    @patch('sys.argv', ['file', 'init'])
    def test_username_password_custom_host_cli_args(self):
        """Test to init with custom host and cli args."""
        sys.argv.append('--user')
        sys.argv.append('myuser')
        sys.argv.append('--password')
        sys.argv.append('pw')
        sys.argv.append('--host')
        sys.argv.append('https://konfuzio.yourcompany.com')
        parser = argparse.ArgumentParser()
        args = parse_args(parser)
        assert credentials(args) == ('myuser', 'pw', 'https://konfuzio.yourcompany.com')

    @patch('builtins.input', side_effect=[''])
    @patch('sys.argv', ['file', 'init'])
    def test_username_password_default_host_cli_args(self, input):
        """Test to init with custom host and cli args."""
        sys.argv.append('--user')
        sys.argv.append('myuser')
        sys.argv.append('--password')
        sys.argv.append('pw')
        parser = argparse.ArgumentParser()
        args = parse_args(parser)
        assert credentials(args) == ('myuser', 'pw', 'https://app.konfuzio.com')

    @patch('builtins.input', side_effect=[''])
    @patch('konfuzio_sdk.cli.getpass', side_effect=['pw'])
    @patch('sys.argv', ['file', 'init'])
    def test_username_no_password_default_host_cli_args(self, input, getpass):
        """Test to init with custom host and cli args."""
        sys.argv.append('--user')
        sys.argv.append('myuser')
        parser = argparse.ArgumentParser()
        args = parse_args(parser)
        assert credentials(args) == ('myuser', 'pw', 'https://app.konfuzio.com')

    @patch('builtins.input', side_effect=['myuser', ''])
    @patch('konfuzio_sdk.cli.getpass', side_effect=['pw'])
    def test_init_project(self, input, getpass):
        """Mock to init a Project without host."""
        with mock.patch('sys.argv', ['file', 'init']):
            with self.assertRaises(PermissionError) as e:
                main()
                assert 'credentials are not correct!' in str(e.exception)

    def test_export_project(self):
        """Test to run CLI."""
        with mock.patch('sys.argv', ['file', 'export_project']):
            with self.assertRaises(SystemExit) as cm:
                main()
        self.assertEqual(cm.exception.code, 2)

    def test_export_project_with_no_int(self):
        """Test to run CLI."""
        with mock.patch('sys.argv', ['file', 'export_project', 'xx']):
            with self.assertRaises(SystemExit) as cm:
                main()
        self.assertEqual(cm.exception.code, 2)

    @patch('konfuzio_sdk.cli.Project.export_project_data', return_value=None)
    def test_export_project_with_int(self, download_function):
        """Test to run CLI."""
        sys.argv = ['file', 'export_project', '46']
        main()
        self.assertTrue(download_function.called)

    @patch('konfuzio_sdk.cli.create_new_project', return_value=None)
    def test_create_project(self, download_function):
        """Test to run CLI."""
        sys.argv = ['file', 'create_project', 'myproject']
        main()
        self.assertTrue(download_function.called)
