"""Generic AI model."""
import abc
import bz2
import logging
import os
import pathlib
import shutil
import sys
from typing import Optional, Union

import bentoml
import cloudpickle
import lz4.frame

from konfuzio_sdk.utils import get_sdk_version, memory_size_of, normalize_memory

logger = logging.getLogger(__name__)


class BaseModel(metaclass=abc.ABCMeta):
    """Base model to define common methods for all AIs."""

    def __init__(self):
        """Initialize a BaseModel class."""
        self.output_dir = None
        self.tokenizer = None
        self.documents = None
        self.test_documents = None
        self.python_version = '.'.join([str(v) for v in sys.version_info[:3]])
        self.konfuzio_sdk_version = get_sdk_version()

    @property
    def name(self):
        """Model class name."""
        return self.__class__.__name__

    def name_lower(self):
        """Convert class name to machine-readable name."""
        return f'{self.name.lower().strip()}'

    @abc.abstractmethod
    def check_is_ready(self):
        """Check if the Model is ready for inference."""

    @property
    @abc.abstractmethod
    def pkl_name(self):
        """Generate a unique extension-less name for a resulting pickle file."""

    @property
    @abc.abstractmethod
    def temp_pkl_file_path(self):
        """Generate a path for temporary pickle file."""

    @property
    @abc.abstractmethod
    def pkl_file_path(self):
        """Generate a path for a resulting pickle file."""

    @property
    @abc.abstractmethod
    def entrypoint_methods(self) -> dict:
        """Create a dict of methods for this class that are exposed via API."""

    @staticmethod
    @abc.abstractmethod
    def has_compatible_interface(other):
        """
        Validate that an instance of an AI implements the same interface defined by this AI class.

        :param other: An instance of an AI to compare with.
        """

    @staticmethod
    def load_model(pickle_path: str, max_ram: Union[None, str] = None):
        """
        Load a previously saved instance of the model.

        :param pickle_path: Path to the pickled model.
        :type pickle_path: str
        :raises FileNotFoundError: If the path is invalid.
        :raises OSError: When the data is corrupted or invalid and cannot be loaded.
        :raises TypeError: When the loaded pickle isn't recognized as a Konfuzio AI model.
        :return: Extraction AI model.
        """
        logger.info(f'Starting loading AI model with path {pickle_path}')

        if not os.path.isfile(pickle_path):
            raise FileNotFoundError('Invalid pickle file path:', pickle_path)

        try:
            if pickle_path.endswith('.pt') or pickle_path.endswith('.pt.lz4'):
                from konfuzio_sdk.trainer.document_categorization import load_categorization_model

                model = load_categorization_model(pickle_path)
            elif pickle_path.endswith('.lz4'):
                with lz4.frame.open(pickle_path, 'rb') as file:
                    model = cloudpickle.load(file)
            else:
                with bz2.open(pickle_path, 'rb') as file:
                    model = cloudpickle.load(file)
        except OSError:
            raise OSError(f'Pickle file {pickle_path} data is invalid.')
        except AttributeError as err:
            if '__forward_module__' in str(err) and '3.9' in sys.version:
                raise AttributeError('Pickle saved with incompatible Python version.') from err
            elif '__forward_is_class__' in str(err) and '3.8' in sys.version:
                raise AttributeError('Pickle saved with incompatible Python version.') from err
            raise
        except ValueError as err:
            if 'unsupported pickle protocol: 5' in str(err) and '3.7' in sys.version:
                raise ValueError('Pickle saved with incompatible Python version.') from err
            raise

        if hasattr(model, 'python_version'):
            logger.info(f'Loaded AI model trained with Python {model.python_version}')
        if hasattr(model, 'konfuzio_sdk_version'):
            logger.info(f'Loaded AI model trained with Konfuzio SDK version {model.konfuzio_sdk_version}')

        max_ram = normalize_memory(max_ram)
        if max_ram and memory_size_of(model) > max_ram:
            logger.error(f"Loaded model's memory use ({memory_size_of(model)}) is greater than max_ram ({max_ram})")

        if not hasattr(model, 'name'):
            raise TypeError('Saved model file needs to be a Konfuzio AbstractExtractionAI instance.')
        elif model.name in {
            'DocumentAnnotationMultiClassModel',
            'DocumentEntityMulticlassModel',
            'SeparateLabelsAnnotationMultiClassModel',
            'SeparateLabelsEntityMultiClassModel',
        }:
            logger.warning(f'Loading legacy {model.name} AI model.')
        else:
            logger.info(f'Loading {model.name} AI model.')

        return model


    def get_sdk_version_for_bento(self):
        """
        Determine the appropriate Konfuzio SDK version or package link for building a Bento file, based on environment variables.

        If running within a pytest environment, return the SDK from a specific Git branch (default: 'master') using the
        'BRANCH_NAME' environment variable. Otherwise, return the stable SDK version constrained to `self.konfuzio_sdk_version`.
        """

        def is_pytest_running():
            return "PYTEST_CURRENT_TEST" in os.environ

        # Determine which version of konfuzio-sdk to use
        if 'BRANCH_NAME' in os.environ:
            branch_name = os.environ['BRANCH_NAME']
        else:
            branch_name = 'master'

        konfuzio_sdk_package = (
            f'https://github.com/konfuzio-ai/konfuzio-sdk/archive/refs/heads/{branch_name}.zip#egg=konfuzio-sdk'
            if is_pytest_running()
            else f'konfuzio-sdk<={self.konfuzio_sdk_version}'
        )
        return konfuzio_sdk_package

    def reduce_model_weight(self):
        """Remove all non-strictly necessary parameters before saving."""
        self.project.lose_weight()
        if (
            self.tokenizer is not None
            and hasattr(self.tokenizer, 'lose_weight')
            and callable(self.tokenizer.lose_weight)
        ):
            self.tokenizer.lose_weight()

    def ensure_model_memory_usage_within_limit(self, max_ram: Optional[str] = None):
        """
        Ensure that a model is not exceeding allowed max_ram.

        :param max_ram: Specify maximum memory usage condition to save model.
        :type max_ram: str
        """
        # if no argument passed, get project max_ram
        if not max_ram and self.project is not None:
            max_ram = self.project.max_ram

        max_ram = normalize_memory(max_ram)

        if max_ram and memory_size_of(self) > max_ram:
            raise MemoryError(f'AI model memory use ({memory_size_of(self)}) exceeds maximum ({max_ram=}).')

    def save(
        self,
        output_dir: str = None,
        include_konfuzio=True,
        reduce_weight=True,
        compression: str = 'lz4',
        keep_documents=False,
        max_ram=None,
    ):
        """
        Save the label model as a compressed pickle object to the release directory.

        Saving is done by: getting the serialized pickle object (via cloudpickle), "optimizing" the serialized object
        with the built-in pickletools.optimize function (see: https://docs.python.org/3/library/pickletools.html),
        saving the optimized serialized object.

        We then compress the pickle file using shutil.copyfileobject which writes in chunks to avoid loading the entire
        pickle file in memory.

        Finally, we delete the cloudpickle file and are left with the compressed pickle file which has a .pkl.lz4 or
        .pkl.bz2 extension.

        For more info on pickle serialization and including dependencies read
        https://github.com/cloudpipe/cloudpickle#overriding-pickles-serialization-mechanism-for-importable-constructs

        :param output_dir: Folder to save AI model in. If None, the default Project folder is used.
        :param include_konfuzio: Enables pickle serialization as a value, not as a reference.
        :param reduce_weight: Remove all non-strictly necessary parameters before saving.
        :param compression: Compression algorithm to use. Default is lz4, bz2 is also supported.
        :param max_ram: Specify maximum memory usage condition to save model.
        :raises MemoryError: When the size of the model in memory is greater than the maximum value.
        :return: Path of the saved model file.
        """
        logger.info('Saving model')
        self.check_is_ready()

        logger.info(f'{output_dir=}')
        logger.info(f'{include_konfuzio=}')
        logger.info(f'{reduce_weight=}')
        logger.info(f'{keep_documents=}')
        logger.info(f'{max_ram=}')
        logger.info(f'{self.konfuzio_sdk_version=}')

        logger.info('Getting save paths')
        if not output_dir:
            self.output_dir = self.project.model_folder
            logger.info(f'new {self.output_dir=}')
        else:
            self.output_dir = output_dir

        # make sure output dir exists
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        temp_pkl_file_path = self.temp_pkl_file_path
        pkl_file_path = self.pkl_file_path

        project_docs = self.project._documents  # to restore project documents after save
        restore_credentials = self.project.credentials
        restore_documents = self.documents
        restore_test_documents = self.test_documents
        self.project.credentials = {}  # to avoid saving credentials
        if reduce_weight:
            self.reduce_model_weight()
        if not keep_documents:
            logger.info('Removing documents before save')
            # to restore Model train and test documents after save
            self.documents = []
            self.test_documents = []

        logger.info(f'Model size: {memory_size_of(self) / 1_000_000} MB')

        try:
            self.ensure_model_memory_usage_within_limit(max_ram)
        except MemoryError as e:
            # restore Documents so that the Project can still be used
            self.project._documents = project_docs
            if not keep_documents:
                self.documents = restore_documents
                self.test_documents = restore_test_documents
            raise e

        sys.setrecursionlimit(999999)

        if include_konfuzio:
            import konfuzio_sdk

            cloudpickle.register_pickle_by_value(konfuzio_sdk)
            # todo register all dependencies?

        if hasattr(self, 'remove_dependencies'):
            self.remove_dependencies()

        # for saving models that have a session as an attribute - otherwise the saving will be unsuccessful
        if hasattr(self, 'client'):
            self.client = None

        logger.info('Saving model with cloudpickle')
        # first save with cloudpickle
        with open(temp_pkl_file_path, 'wb') as f:  # see: https://stackoverflow.com/a/9519016/5344492
            cloudpickle.dump(self, f)

        if compression == 'lz4':
            logger.info('Compressing model with lz4')
            pkl_file_path += '.lz4'
            # then save to lz4 in chunks
            with open(temp_pkl_file_path, 'rb') as input_f:
                with lz4.frame.open(pkl_file_path, 'wb') as output_f:
                    shutil.copyfileobj(input_f, output_f)
        elif compression == 'bz2':
            logger.info('Compressing model with bz2')
            pkl_file_path += '.bz2'
            # then save to bz2 in chunks
            with open(temp_pkl_file_path, 'rb') as input_f:
                with bz2.open(pkl_file_path, 'wb') as output_f:
                    shutil.copyfileobj(input_f, output_f)
        else:
            raise ValueError(f'Unknown compression algorithm: {compression}')

        logger.info('Deleting temporary cloudpickle file')
        # then delete cloudpickle file
        os.remove(temp_pkl_file_path)

        size_string = f'{os.path.getsize(pkl_file_path) / 1_000_000} MB'
        logger.info(f'Model ({size_string}) {self.name_lower()} was saved to {pkl_file_path}')

        # restore Documents of the Category and Model so that we can continue using them as before
        if not keep_documents:
            self.documents = restore_documents
            self.test_documents = restore_test_documents
        self.project._documents = project_docs
        self.project.credentials = restore_credentials

        return pkl_file_path

    def save_bento(self, build=True, output_dir=None) -> Union[None, tuple]:
        """
        Save AI as a BentoML model in the local store.

        :param build: Bundle the model into a BentoML service and store it in the local store.
        :param output_dir: If present, a .bento archive will also be saved to this directory.

        :return: None if build=False, otherwise a tuple of (saved_bento, archive_path).
        """
        # get the current directory to restore later
        working_dir = os.getcwd()

        if output_dir and not build:
            raise ValueError('Cannot specify output_dir without build=True')

        # cache the pickle name to avoid changing it during the save process (as it includes timestamps)
        self._pkl_name = self.pkl_name

        saved_model = bentoml.picklable_model.save_model(
            name=self._pkl_name,
            model=self,
            signatures=self.entrypoint_methods,
            metadata=self.bento_metadata,
        )
        logger.info(f'Model saved in the local BentoML store: {saved_model}')

        if not build:
            # restore the working directory
            # see https://github.com/bentoml/BentoML/issues/3403
            os.chdir(working_dir)
            return

        saved_bento = self.build_bento(bento_model=saved_model)
        logger.info(f'Bento created: {saved_bento}')

        if not output_dir:
            # restore the working directory
            os.chdir(working_dir)
            return saved_bento, None  # None = no archive saved

        archive_path = saved_bento.export(output_dir)
        logger.info(f'Bento archive saved: {archive_path}')

        # restore the working directory
        os.chdir(working_dir)
        return saved_bento, archive_path


def load_bento(model_name):
    """Load AI as a Bento ML instance."""
    return bentoml.picklable_model.load_model(model_name)
