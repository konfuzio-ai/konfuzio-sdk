"""Initialize AI-related dependencies safely."""
import logging

from typing import List

from konfuzio_sdk.settings_importer import OPTIONAL_IMPORT_ERROR

logger = logging.getLogger(__name__)


class PackageWrapper:
    """Heavy dependencies are encapsulated and handled if they are not part of the lightweight SDK installation."""

    def __init__(self, package_name: str, required_for_modules: List[str]):
        """
        Initialize the wrapper.

        :param package_name: Name of a package to be wrapped.
        :param required_for_modules: A list of modules the package is required for.
        """
        self.package_name = package_name
        self.required_for_modules = ', '.join(required_for_modules) if required_for_modules else ''
        self.package = self._import_package()

    def _import_package(self):
        """Import the package if it is installed, throw an error if it is not."""
        from konfuzio_sdk.settings_importer import DO_NOT_LOG_IMPORT_ERRORS

        try:
            package = __import__(self.package_name)
            return package
        except ImportError:
            if DO_NOT_LOG_IMPORT_ERRORS:
                pass
            else:
                raise logging.error(
                    OPTIONAL_IMPORT_ERROR.replace('*modulename*', self.package_name)
                    + (
                        f" This library is required for modules: \
                                    {self.required_for_modules}."
                        if self.required_for_modules
                        else ''
                    )
                )

    def __getattr__(self, item):
        """Import a method of a package."""
        if self.package:
            return getattr(self.package, item)
        else:
            raise ImportError(f"The '{self.package_name}' library is missing. Please install it.")


spacy = PackageWrapper('spacy', ['Document Categorization AI'])
tensorflow = PackageWrapper('tensorflow', ['File Splitting AI'])
timm = PackageWrapper('timm', ['Document Categorization AI'])
torch = PackageWrapper('torch', ['Document Categorization AI, File Splitting AI'])
torchvision = PackageWrapper('torchvision', ['Document Categorization AI'])
transformers = PackageWrapper('transformers', ['Document Categorization AI, File Splitting AI'])
