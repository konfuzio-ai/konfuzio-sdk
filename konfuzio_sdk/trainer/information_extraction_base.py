import abc
import bz2
import logging
import os
import pathlib
import shutil
import sys

from copy import deepcopy
from inspect import signature
from typing import Tuple, Optional, List, Dict

import numpy
import pandas
import cloudpickle

from konfuzio_sdk.data import Document, Annotation, Category, AnnotationSet, Label, LabelSet, Span

from konfuzio_sdk.normalize import (
    normalize_to_float,
    normalize_to_date,
    normalize_to_percentage,
    normalize_to_positive_float,
)
from konfuzio_sdk.utils import (
    normalize_memory,
    get_sdk_version,
    memory_size_of,
)

logger = logging.getLogger(__name__)


class BaseModel(metaclass=abc.ABCMeta):
    """Base model to define common methods for all AIs."""

    def __init__(self):
        """Initialize a BaseModel class."""
        self.output_dir = None
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
    def temp_pkl_file_path(self):
        """Generate a path for temporary pickle file."""

    @property
    @abc.abstractmethod
    def pkl_file_path(self):
        """Generate a path for a resulting pickle file."""

    @staticmethod
    @abc.abstractmethod
    def has_compatible_interface(other):
        """
        Validate that an instance of an AI implements the same interface defined by this AI class.

        :param other: An instance of an AI to compare with.
        """

    def reduce_model_weight(self):
        """Remove all non-strictly necessary parameters before saving."""
        self.project.lose_weight()
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
            raise MemoryError(f"AI model memory use ({memory_size_of(self)}) exceeds maximum ({max_ram=}).")

    def save(
            self, output_dir: str = None, include_konfuzio=True, reduce_weight=True, keep_documents=False, max_ram=None
    ):
        """
        Save the label model as bz2 compressed pickle object to the release directory.

        Saving is done by: getting the serialized pickle object (via cloudpickle), "optimizing" the serialized object
        with the built-in pickletools.optimize function (see: https://docs.python.org/3/library/pickletools.html),
        saving the optimized serialized object.

        We then compress the pickle file with bz2 using shutil.copyfileobject which writes in chunks to avoid loading
        the entire pickle file in memory.

        Finally, we delete the cloudpickle file and are left with the bz2 file which has a .pkl extension.

        :param output_dir: Folder to save AI model in. If None, the default Project folder is used.
        :param include_konfuzio: Enables pickle serialization as a value, not as a reference (for more info, read
        https://github.com/cloudpipe/cloudpickle#overriding-pickles-serialization-mechanism-for-importable-constructs).
        :param reduce_weight: Remove all non-strictly necessary parameters before saving.
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

        if reduce_weight:
            project_docs = self.project._documents  # to restore project documents after save
            self.reduce_model_weight()
        if not keep_documents:
            logger.info('Removing documents before save')
            # to restore Model train and test documents after save
            restore_documents = self.documents
            restore_test_documents = self.test_documents
            self.documents = []
            self.test_documents = []

        logger.info(f'Model size: {memory_size_of(self) / 1_000_000} MB')

        self.ensure_model_memory_usage_within_limit(max_ram)

        sys.setrecursionlimit(999999)

        if include_konfuzio:
            import konfuzio_sdk

            cloudpickle.register_pickle_by_value(konfuzio_sdk)
            # todo register all dependencies?

        logger.info('Saving model with cloudpickle')
        # first save with cloudpickle
        with open(temp_pkl_file_path, 'wb') as f:  # see: https://stackoverflow.com/a/9519016/5344492
            cloudpickle.dump(self, f)

        logger.info('Compressing model with bz2')

        # then save to bz2 in chunks
        with open(temp_pkl_file_path, 'rb') as input_f:
            with bz2.open(pkl_file_path, 'wb') as output_f:
                shutil.copyfileobj(input_f, output_f)

        logger.info('Deleting cloudpickle file')
        # then delete cloudpickle file
        os.remove(temp_pkl_file_path)

        size_string = f'{os.path.getsize(pkl_file_path) / 1_000_000} MB'
        logger.info(f'Model ({size_string}) {self.name_lower()} was saved to {pkl_file_path}')

        # restore Documents of the Category and Model so that we can continue using them as before
        if not keep_documents:
            self.documents = restore_documents
            self.test_documents = restore_test_documents
        if reduce_weight:
            self.project._documents = project_docs

        return pkl_file_path


class Trainer(BaseModel):
    """Parent class for all Extraction AIs, to extract information from unstructured human readable text."""

    def __init__(self, category: Category, *args, **kwargs):
        """Initialize ExtractionModel."""
        # Go through keyword arguments, and either save their values to our
        # instance, or raise an error.
        super().__init__()
        self.category = category
        self.clf = None
        self.label_feature_list = None  # will be set later

        self.df_train = None

        self.evaluation = None

    def fit(self):
        """Use as placeholder Function."""
        logger.warning(f'{self} does not train a classifier.')
        pass

    def evaluate(self):
        """Use as placeholder Function."""
        logger.warning(f'{self} does not evaluate results.')
        pass

    def extract(self):
        """Use as placeholder Function."""
        logger.warning(f'{self} does not extract.')
        pass

    def extraction_result_to_document(self, document: Document, extraction_result: dict) -> Document:
        """Return a virtual Document annotated with AI Model output."""
        virtual_doc = deepcopy(document)
        virtual_annotation_set_id = 1  # counter for across mult. Annotation Set groups of a Label Set

        # define Annotation Set for the Category Label Set: todo: this is unclear from API side
        # default Annotation Set will be always added even if there are no predictions for it
        category_label_set = self.category.project.get_label_set_by_id(self.category.id_)
        virtual_default_annotation_set = AnnotationSet(
            document=virtual_doc, label_set=category_label_set, id_=virtual_annotation_set_id
        )
        virtual_annotation_set_id += 1
        for label_or_label_set_name, information in extraction_result.items():

            if isinstance(information, pandas.DataFrame):
                if information.empty:
                    continue

                # annotations belong to the default Annotation Set
                label = self.category.project.get_label_by_name(label_or_label_set_name)
                self.add_extractions_as_annotations(
                    document=virtual_doc,
                    extractions=information,
                    label=label,
                    label_set=category_label_set,
                    annotation_set=virtual_default_annotation_set,
                )
            # process multi Annotation Sets that are not part of the category Label Set
            else:
                label_set = self.category.project.get_label_set_by_name(label_or_label_set_name)

                if not isinstance(information, list):
                    information = [information]

                for entry in information:  # represents one of pot. multiple annotation-sets belonging of one LabelSet
                    if label_set is not category_label_set:
                        virtual_annotation_set = AnnotationSet(
                            document=virtual_doc, label_set=label_set, id_=virtual_annotation_set_id
                        )
                        virtual_annotation_set_id += 1
                    else:
                        virtual_annotation_set = virtual_default_annotation_set

                    for label_name, extractions in entry.items():
                        label = self.category.project.get_label_by_name(label_name)
                        self.add_extractions_as_annotations(
                            document=virtual_doc,
                            extractions=extractions,
                            label=label,
                            label_set=label_set,
                            annotation_set=virtual_annotation_set,
                        )

        return virtual_doc

    @staticmethod
    def add_extractions_as_annotations(
            extractions: pandas.DataFrame,
            document: Document,
            label: Label,
            label_set: LabelSet,
            annotation_set: AnnotationSet,
    ) -> None:
        """Add the extraction of a model to the document."""
        if not isinstance(extractions, pandas.DataFrame):
            raise TypeError(f'Provided extraction object should be a Dataframe, got a {type(extractions)} instead')
        if not extractions.empty:
            # TODO: define required fields
            required_fields = ['start_offset', 'end_offset', 'confidence']
            if not set(required_fields).issubset(extractions.columns):
                raise ValueError(
                    f'Extraction do not contain all required fields: {required_fields}.'
                    f' Extraction columns: {extractions.columns.to_list()}'
                )

            extracted_spans = extractions[required_fields].sort_values(by='confidence', ascending=False)

            for span in extracted_spans.to_dict('records'):  # todo: are start_offset and end_offset always ints?
                annotation = Annotation(
                    document=document,
                    label=label,
                    confidence=span['confidence'],
                    label_set=label_set,
                    annotation_set=annotation_set,
                    spans=[Span(start_offset=span['start_offset'], end_offset=span['end_offset'])],
                )
                if annotation.spans[0].offset_string is None:
                    raise NotImplementedError(
                        f"Extracted {annotation} does not have a correspondence in the " f"text of {document}."
                    )

    @classmethod
    def merge_horizontal(cls, res_dict: Dict, doc_text: str) -> Dict:
        """Merge contiguous spans with same predicted label.

        See more details at https://dev.konfuzio.com/sdk/explanations.html#horizontal-merge
        """
        logger.info("Horizontal merge.")
        merged_res_dict = dict()  # stores final results
        for label, items in res_dict.items():
            res_dicts = []
            buffer = []
            end = None

            for _, row in items.iterrows():  # iterate over the rows in the DataFrame
                # if they are valid merges then add to buffer
                if end and cls.is_valid_horizontal_merge(row, buffer, doc_text):
                    buffer.append(row)
                    end = row['end_offset']
                else:  # else, flush the buffer by creating a res_dict
                    if buffer:
                        res_dict = cls.flush_buffer(buffer, doc_text)
                        res_dicts.append(res_dict)
                    buffer = []
                    buffer.append(row)
                    end = row['end_offset']
            if buffer:  # flush buffer at the very end to clear anything left over
                res_dict = cls.flush_buffer(buffer, doc_text)
                res_dicts.append(res_dict)
            merged_df = pandas.DataFrame(
                res_dicts
            )  # convert the list of res_dicts created by `flush_buffer` into a DataFrame

            merged_res_dict[label] = merged_df

        return merged_res_dict

    @staticmethod
    def flush_buffer(buffer: List[pandas.Series], doc_text: str) -> Dict:
        """
        Merge a buffer of entities into a dictionary (which will eventually be turned into a DataFrame).

        A buffer is a list of pandas.Series objects.
        """
        assert 'label_name' in buffer[0]
        label = buffer[0]['label_name']

        starts = buffer[0]['start_offset']
        ends = buffer[-1]['end_offset']
        text = doc_text[starts:ends]

        res_dict = dict()
        res_dict['start_offset'] = starts
        res_dict['end_offset'] = ends
        res_dict['label_name'] = label
        res_dict['offset_string'] = text
        res_dict['confidence'] = numpy.mean([b['confidence'] for b in buffer])
        return res_dict

    @staticmethod
    def is_valid_horizontal_merge(
            row: pandas.Series,
            buffer: List[pandas.Series],
            doc_text: str,
            max_offset_distance: int = 5,
    ) -> bool:
        """
        Verify if the merging that we are trying to do is valid.

        A merging is valid only if:
          * All spans have the same predicted Label
          * Confidence of predicted Label is above the Label threshold
          * All spans are on the same line
          * No extraneous characters in between spans
          * A maximum of 5 spaces in between spans
          * The Label type is not one of the following: 'Number', 'Positive Number', 'Percentage', 'Date'
            OR the resulting merging create a span normalizable to the same type

        :param row: Row candidate to be merged to what is already in the buffer.
        :param buffer: Previous information.
        :param doc_text: Text of the document.
        :param max_offset_distance: Maximum distance between two entities that can be merged.
        :return: If the merge is valid or not.
        """
        if row['confidence'] < row['label_threshold']:
            return False

        # sanity checks
        if buffer[-1]['label_name'] != row['label_name']:
            return False
        elif buffer[-1]['confidence'] < buffer[-1]['label_threshold']:
            return False

        # Do not merge if any character in between the two Spans
        if not all([c == ' ' for c in doc_text[buffer[-1]['end_offset']: row['start_offset']]]):
            return False

        # Do not merge if the difference in the offsets is bigger than the maximum offset distance
        if row['start_offset'] - buffer[-1]['end_offset'] > max_offset_distance:
            return False

        # only merge if text is on same line
        if '\n' in doc_text[buffer[0]['start_offset']: row['end_offset']]:
            return False

        # Do not merge overlapping spans
        if row['start_offset'] < buffer[-1]['end_offset']:
            return False

        data_type = row['data_type']
        # always merge if not one of these data types
        if data_type not in {'Number', 'Positive Number', 'Percentage', 'Date'}:
            return True

        merge = None
        text = doc_text[buffer[0]['start_offset']: row['end_offset']]

        # only merge percentages/dates/(positive) numbers if the result is still normalizable to the type
        if data_type == 'Percentage':
            merge = normalize_to_percentage(text)
        elif data_type == 'Date':
            merge = normalize_to_date(text)
        elif data_type == 'Number':
            merge = normalize_to_float(text)
        elif data_type == 'Positive Number':
            merge = normalize_to_positive_float(text)

        return merge is not None

    @staticmethod
    def has_compatible_interface(other) -> bool:
        """
        Validate that an instance of an Extraction AI implements the same interface as Trainer.

        An Extraction AI should implement methods with the same signature as:
        - Trainer.__init__
        - Trainer.fit
        - Trainer.extract
        - Trainer.check_is_ready

        :param other: An instance of an Extraction AI to compare with.
        """
        try:
            return (
                    signature(other.__init__).parameters['category'].annotation.__name__ == 'Category'
                    and signature(other.extract).parameters['document'].annotation.__name__ == 'Document'
                    and signature(other.extract).return_annotation.__name__ == 'Document'
                    and signature(other.fit)
                    and signature(other.check_is_ready)
            )
        except KeyError:
            return False
        except AttributeError:
            return False
