"""Test the module for wrapping the extra dependencies."""
import pytest
import sklearn

from konfuzio_sdk.extras import PackageWrapper


def test_package_wrapper():
    """Test the PackageWrapper class."""
    from konfuzio_sdk.settings_importer import DO_NOT_LOG_IMPORT_ERRORS

    packaged = PackageWrapper('sklearn', ['SDK'])
    assert packaged.package_name == 'sklearn'
    assert packaged.required_for_modules == 'SDK'
    assert packaged.package == sklearn
    assert DO_NOT_LOG_IMPORT_ERRORS
    missing_package = PackageWrapper('samplepackage')
    assert not missing_package.package
    with pytest.raises(ImportError, match='library is missing'):
        missing_package.method()


def test_is_dependency_installed():
    """Test method for checking if a package is installed."""
    from konfuzio_sdk.settings_importer import is_dependency_installed

    result = is_dependency_installed('sklearn')
    assert result
    result = is_dependency_installed('samplepackage')
    assert not result
