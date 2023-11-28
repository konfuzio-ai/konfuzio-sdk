"""Validate CLI functions."""
import sys
from unittest import mock, TestCase
from unittest.mock import patch

from konfuzio_sdk.cli import CLI_ERROR, main, credentials


class TestCLI(TestCase):
    """Test the konfuzio_sdk CLI."""

    def test_cli_error(self):
        """Test if CLI print is available."""
        assert "Please enter a valid command line option." in CLI_ERROR

    def test_help(self):
        """Test to run CLI."""
        with mock.patch('sys.argv', ['file', '--help']):
            assert main() == -1

    def test_without_input(self):
        """Test to run CLI."""
        with mock.patch('sys.argv', ['file', None]):
            assert main() == -1

    @patch('builtins.input', side_effect=['myuser', 'https://konfuzio.yourcompany.com'])
    @patch("konfuzio_sdk.cli.getpass", side_effect=['pw'])
    def test_username_password_custom_host(self, input, getpass):
        """Test to init with custom Host. See https://stackoverflow.com/a/55580216 and 56401696 for patches."""
        assert credentials() == ("myuser", "pw", "https://konfuzio.yourcompany.com")

    @patch('builtins.input', side_effect=['myuser', ''])
    @patch("konfuzio_sdk.cli.getpass", side_effect=['pw'])
    def test_username_password_default_host(self, input, getpass):
        """Test to init with default Host."""
        assert credentials() == ("myuser", "pw", "https://app.konfuzio.com")

    @patch('builtins.input', side_effect=['myuser', 'https://konfuzio.yourcompany.com//'])
    @patch("konfuzio_sdk.cli.getpass", side_effect=['pw'])
    def test_removing_trailing_slashes(self, input, getpass):
        """Test to ensure that any trailing slash(es) in the end of the host URL are removed."""
        assert credentials() == ("myuser", "pw", "https://konfuzio.yourcompany.com")

    @patch('builtins.input', side_effect=['myuser', ''])
    @patch("konfuzio_sdk.cli.getpass", side_effect=['pw'])
    def test_init_project(self, input, getpass):
        """Mock to init a Project without host."""
        with mock.patch('sys.argv', ['file', 'init']):
            with self.assertRaises(PermissionError) as e:
                main()
                assert 'credentials are not correct!' in str(e.exception)

    def test_export_project(self):
        """Test to run CLI."""
        with mock.patch('sys.argv', ['file', 'export_project']):
            assert main() == -1

    def test_export_project_with_no_int(self):
        """Test to run CLI."""
        with mock.patch('sys.argv', ['file', 'export_project', 'xx']):
            assert main() == -1

    @patch("konfuzio_sdk.cli.Project.export_project_data", return_value=None)
    def test_export_project_with_int(self, download_function):
        """Test to run CLI."""
        sys.argv = ['file', 'export_project', '46']
        main()
        self.assertTrue(download_function.called)

    @patch("konfuzio_sdk.cli.create_new_project", return_value=None)
    def test_create_project(self, download_function):
        """Test to run CLI."""
        sys.argv = ['file', 'create_project', 'myproject']
        main()
        self.assertTrue(download_function.called)
