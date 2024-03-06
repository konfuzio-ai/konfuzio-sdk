"""Initialize AI-related dependencies safely."""
import abc
import functools
import logging
from typing import List

from konfuzio_sdk.settings_importer import OPTIONAL_IMPORT_ERROR

logger = logging.getLogger(__name__)


class PackageWrapper:
    """Heavy dependencies are encapsulated and handled if they are not part of the lightweight SDK installation."""

    def __init__(self, package_name: str, required_for_modules: List[str] = None):
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
                        f' This library is required for modules: \
                                    {self.required_for_modules}.'
                        if self.required_for_modules
                        else ''
                    )
                )

    def __getattr__(self, item):
        """Import a method of a package."""
        from konfuzio_sdk.settings_importer import DO_NOT_LOG_IMPORT_ERRORS

        if self.package:
            return getattr(self.package, item)
        else:
            if DO_NOT_LOG_IMPORT_ERRORS:
                pass
            else:
                raise ImportError(f"The '{self.package_name}' library is missing. Please install it.")


class ModuleWrapper:
    """Handle missing dependencies' classes to avoid metaclass conflict."""

    def __init__(self, module: str):
        """Initialize the wrapper."""
        self._replace(module)

    def _replace(self, module):
        """Replace the original class with the placeholder."""
        self.replaced = type(module, (object,), {'__metaclass__': abc.ABCMeta})


datasets = PackageWrapper('datasets', ['File Splitting AI'])
evaluate = PackageWrapper('evaluate', ['File Splitting AI'])
mlflow = PackageWrapper('mlflow', ['File Splitting AI'])  #!TODO: Add to other AI types when needed
spacy = PackageWrapper('spacy', ['Document Categorization AI'])
if spacy.package:
    SpacyPhraseMatcher = spacy.matcher.PhraseMatcher
    SpacyLanguage = spacy.language.Language
else:
    SpacyPhraseMatcher = ModuleWrapper('SpacyPhraseMatcher').replaced
    SpacyLanguage = ModuleWrapper('SpacyLanguage').replaced
tensorflow = PackageWrapper('tensorflow', ['File Splitting AI'])
timm = PackageWrapper('timm', ['Document Categorization AI'])
torch = PackageWrapper('torch', ['Document Categorization AI, File Splitting AI'])
if torch.package:
    Module = torch.nn.Module
    Tensor = torch.Tensor
    FloatTensor = torch.FloatTensor
    Optimizer = torch.optim.Optimizer
    DataLoader = torch.utils.data.DataLoader
    LongTensor = torch.LongTensor
else:
    Module = ModuleWrapper('Module').replaced
    Tensor = ModuleWrapper('Tensor').replaced
    FloatTensor = ModuleWrapper('FloatTensor').replaced
    Optimizer = ModuleWrapper('Optimizer').replaced
    DataLoader = ModuleWrapper('DataLoader').replaced
    LongTensor = ModuleWrapper('LongTensor').replaced
torchvision = PackageWrapper('torchvision', ['Document Categorization AI'])
transformers = PackageWrapper('transformers', ['Document Categorization AI, File Splitting AI'])
if transformers.package:
    Trainer = transformers.Trainer
    TrainerCallback = transformers.TrainerCallback
else:
    Trainer = ModuleWrapper('Trainer').replaced
    TrainerCallback = ModuleWrapper('TrainerCallback').replaced


def torch_no_grad(method):
    """Wrap the decorator."""
    if torch.package:

        @torch.no_grad()
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            result = method(*args, **kwargs)
            return result

        return wrapper

    else:
        return method
