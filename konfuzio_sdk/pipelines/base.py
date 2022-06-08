"""ExtractionModels to extract information from unstructured text."""
import bz2
import datetime
import logging
import os
import pathlib
import shutil
import sys

import cloudpickle
from konfuzio_sdk import BASE_DIR
from konfuzio_sdk.data import Document
from konfuzio_sdk.evaluate import compare
from konfuzio_sdk.pipelines.load_data import get_latest_document_model, load_pickle, extraction_result_to_document
from konfuzio_sdk.utils import get_timestamp

#  extraction_result_to_document

logger = logging.getLogger(__name__)


class ExtractionModel:
    """Base Model to extract information from unstructured human readable text."""

    def __init__(self):
        """Initialize ExtractionModel."""
        # Go through keyword arguments, and either save their values to our
        # instance, or raise an error.
        self.clf = None
        self.name = self.__class__.__name__
        self.label_feature_list = None  # will be set later

        self.df_data = None
        self.df_valid = None
        self.df_train = None
        self.df_test = None

        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None

        self.created_at = datetime.datetime.now()
        self.updated_at = self.created_at

        # for key, value in kwargs.items():
        #     setattr(self, key, value)

    def build(self):
        """Build an ExtractionModel."""
        self.tokenize(documents=self.documents + self.test_documents)
        self.create_candidates_dataset()
        self.train_valid_split()
        self.fit()
        self.fit_label_set_clf()
        if not self.df_test.empty:
            self.evaluate()
        else:
            logger.error('The test set is empty. Skip evaluation for test data.')
        self.lose_weight()
        return self

    def name_lower(self):
        """Convert class name to machine readable name."""
        return f'{self.name.lower().strip()}'

    def lose_weight(self):
        """Delete everything that is not necessary for extraction."""
        self.df_valid = None
        self.df_train = None
        self.df_test = None

        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None

        # TODO what is this?
        self.valid_data = None
        self.training_data = None
        self.test_data = None
        self.df_data_list = None

        logger.info(f'Lose weight was executed on {self.name}')

    def get_ai_model(self):
        """Try to load the latest pickled model."""
        try:
            return load_pickle(get_latest_document_model(f'*_{self.name_lower()}.pkl'))
        except FileNotFoundError:
            return None

    def tokenize(self, documents, multiprocess):
        """Use as placeholder Function."""
        logger.warning(f'{self} does not tokenize.')
        pass

    def create_candidates_dataset(self):
        """Use as placeholder Function."""
        logger.warning(f'{self} does not train a classifier.')
        pass

    def train_valid_split(self):
        """Use as placeholder Function."""
        logger.warning(f'{self} does not use a valid and train data split.')
        pass

    def fit(self):
        """Use as placeholder Function."""
        logger.warning(f'{self} does not train a classifier.')
        pass

    def fit_label_set_clf(self):
        """Use as placeholder Function."""
        logger.warning(f'{self} does not train a label set classifier.')
        pass

    def evaluate(self):
        """Use as placeholder Function."""
        logger.warning(f'{self} does not evaluate results.')
        pass

    # the extract function must accept arbitrary args and kwargs to support multiple server versions
    def extract(self, *args, **kwargs):
        """Use as placeholder Function."""
        logger.warning(f'{self} does not extract.')
        pass

    def evaluate_extraction_model(self, document: Document):
        """Run and evaluate model on the document."""
        # build the doc from model results
        extraction_result = self.extract(text=document.text, bbox=document.get_bbox(), pages=document.pages)
        virtual_doc = extraction_result_to_document(document, extraction_result)

        # Tokenize document here in order to be able to evaluate the tokenizer.
        self.tokenize([virtual_doc], multiprocess=False)

        return compare(document, virtual_doc)


    def save(self, name=None, output_dir=None, include_konfuzio=True):
        """
        Save the label model as bz2 compressed pickle object to the release directory.

        Saving is done by: getting the serialized pickle object (via dill), "optimizing" the serialized object with the
        built-in pickletools.optimize function (see: https://docs.python.org/3/library/pickletools.html), saving the
        optimized serialized object.

        We then compress the pickle file with bz2 using shutil.copyfileobject which writes in chunks to avoid loading
        the entire pickle file in memory.

        Finally, we delete the dill file and are left with the bz2 file which has a .pkl extension.

        :return: Path of the saved model file
        """
        # Keep Documents of the Category so that we can restore them later
        category_documents = self.category.documents() + self.category.test_documents()

        # TODO: add Document.lose_weight in SDK - remove NO_LABEL Annotations from the Documents
        for document in category_documents:
            no_label_annotations = document.annotations(label=self.no_label)
            clean_annotations = list(set(document.annotations()) - set(no_label_annotations))
            document._annotations = clean_annotations

        self.lose_weight()

        from pympler import asizeof
        logger.info(f'Saving model - {asizeof.asizeof(self) / 1_000_000} MB')

        sys.setrecursionlimit(99999999)

        logger.info('Getting save paths')
        import konfuzio_sdk
        import konfuzio

        if include_konfuzio:
            cloudpickle.register_pickle_by_value(konfuzio_sdk)
            cloudpickle.register_pickle_by_value(konfuzio)

        if not name:
            name = self.category.name.lower()
        if not output_dir:
            output_dir = BASE_DIR
        self.updated_at = datetime.datetime.now()  # TODO statement has no effect
        # moke sure output dir exists
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(output_dir, f'{get_timestamp()}_{name}')
        temp_pkl_file_path = file_path + '.dill'
        pkl_file_path = file_path + '.pkl'
        try:
            # see: https://stackoverflow.com/a/9519016/5344492

            logger.info('Saving model with dill')

            # first save with dill
            with open(temp_pkl_file_path, 'wb') as f:
                cloudpickle.dump(self, f)

            logger.info('Compressing model with bz2')

            # then save to bz2 in chunks
            with open(temp_pkl_file_path, 'rb') as input_f:
                with bz2.open(pkl_file_path, 'wb') as output_f:
                    shutil.copyfileobj(input_f, output_f)

            logger.info('Deleting dill file')

            # then delete dill file
            os.remove(temp_pkl_file_path)

            size_string = f'{os.path.getsize(pkl_file_path) / 1_000_000} MB'
            logger.info(f'Model ({size_string}) {name} was saved to {pkl_file_path}')
        except AttributeError:
            logger.exception('Cannot save pickled object.')

        # restore Documents of the Category so that we can run the evaluation later
        self.category.project._documents = category_documents

        return pkl_file_path

