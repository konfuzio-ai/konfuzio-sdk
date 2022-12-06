"""Find similarities between Documents or Pages via comparison between their texts."""
import abc
import bz2
import cloudpickle
import konfuzio_sdk
import logging
import os
import pathlib
import shutil
import sys

from copy import deepcopy
from typing import List, Tuple

from konfuzio_sdk.data import Document, Page
from konfuzio_sdk.trainer.information_extraction import load_model
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.samples import LocalTextProject
from konfuzio_sdk.utils import get_timestamp

logger = logging.getLogger(__name__)


class AbstractFileSplittingModel(metaclass=abc.ABCMeta):
    """Abstract class for the filesplitting model."""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the class."""

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the custom model on the training Documents."""

    @abc.abstractmethod
    def save(self, model_path=""):
        """
        Save the trained model.

        :param model_path: Path to save the model to.
        :type model_path: str
        """

    @abc.abstractmethod
    def predict(self, page: Page) -> Page:
        """
        Take a Page as an input and return 1 for first page and 0 for not first page.

        :param page: A Page to label first or non-first.
        :type page: Page
        :return: A Page with or without is_first_page label.
        """


class FileSplittingEvaluation:
    """Evaluate the quality of the filesplitting logic."""

    def __init__(
        self, documents: List[Tuple[Document, Document]], calculate_by_category: bool = False, allow_zero: bool = False
    ):
        """
        Initialize and run the metrics calculation.

        :param documents: A list of Document pairs – first one is ground truth, second is the prediction.
        :type documents: list
        :param calculate_by_category: Calculate metrics for each Category independently.
        :type calculate_by_category: bool
        :param allow_zero: If true, will calculate None for precision and recall when the straightforward application
        of the formula would otherwise result in 0/0. Raises ZeroDivisionError otherwise.
        :type allow_zero: bool
        """
        self.documents = documents
        self.calculate_by_category = calculate_by_category
        self.allow_zero = allow_zero
        if self.calculate_by_category:
            self.calculate_by_category()
        else:
            self.calculate()

    def calculate(self):
        """Calculate metrics for the filesplitting logic."""
        tp = 0
        fp = 0
        fn = 0
        for ground_truth, prediction in self.documents:
            for page_gt, page_pr in zip(ground_truth.pages(), prediction.pages()):
                if page_gt.number == 1 and hasattr(page_pr, 'is_first_page'):
                    tp += 1
                elif page_gt.number > 1 and hasattr(page_pr, 'is_first_page'):
                    fp += 1
                elif page_gt.number == 1 and not hasattr(page_pr, 'is_first_page'):
                    fn += 1
        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            if self.allow_zero:
                precision = None
            else:
                raise ZeroDivisionError(
                    "TP and FP are zero, please specify allow_zero=True if you want precision to be None."
                )
        if tp + fn != 0:
            recall = tp / (tp + fn)
        else:
            if self.allow_zero:
                recall = None
            else:
                raise ZeroDivisionError(
                    "TP and FN are zero, please specify allow_zero=True if you want recall to be None."
                )
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            if self.allow_zero:
                f1 = None
            else:
                raise ZeroDivisionError("FP and FN are zero, please specify allow_zero=True if you want F1 to be None.")
        self.evaluation_results = {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}

    def calculate_by_category(self):
        """Calculate metrics by Category independently."""
        categories = list(set([doc_pair[0].category for doc_pair in self.documents]))
        self.evaluation_results = {'tp': {}, 'fp': {}, 'fn': {}, 'precision': {}, 'recall': {}, 'f1': {}}
        for category in categories:
            tp = 0
            fp = 0
            fn = 0
            for ground_truth, prediction in [
                document for document in self.documents if document.category.id_ == category.id_
            ]:
                for page_gt, page_pr in zip(ground_truth.pages(), prediction.pages()):
                    if page_gt.number == 1 and hasattr(page_pr, 'is_first_page'):
                        tp += 1
                    elif page_gt.number > 1 and hasattr(page_pr, 'is_first_page'):
                        fp += 1
                    elif page_gt.number == 1 and not hasattr(page_pr, 'is_first_page'):
                        fn += 1
            if tp + fp != 0:
                precision = tp / (tp + fp)
            else:
                if self.allow_zero:
                    precision = None
                else:
                    raise ZeroDivisionError(
                        "TP and FP are zero, please specify allow_zero=True if you want precision to be None."
                    )
            if tp + fn != 0:
                recall = tp / (tp + fn)
            else:
                if self.allow_zero:
                    recall = None
                else:
                    raise ZeroDivisionError(
                        "TP and FN are zero, please specify allow_zero=True if you want recall to be None."
                    )
            if precision + recall != 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                if self.allow_zero:
                    f1 = None
                else:
                    raise ZeroDivisionError(
                        "FP and FN are zero, please specify allow_zero=True if you want F1 to be None."
                    )
            self.evaluation_results['tp'][category.id_] = tp
            self.evaluation_results['fp'][category.id_] = fp
            self.evaluation_results['fn'][category.id_] = fn
            self.evaluation_results['precision'][category.id_] = precision
            self.evaluation_results['recall'][category.id_] = recall
            self.evaluation_results['f1'][category.id_] = f1

    @property
    def tp(self):
        """Return correctly predicted first Pages."""
        return self.evaluation_results['tp']

    @property
    def fp(self):
        """Return non-first Pages incorrectly predicted as first."""
        return self.evaluation_results['fp']

    @property
    def fn(self):
        """Return first Pages incorrectly predicted as non-first."""
        return self.evaluation_results['fn']

    @property
    def precision(self):
        """Return precision."""
        return self.evaluation_results['precision']

    @property
    def recall(self):
        """Return recall."""
        return self.evaluation_results['recall']

    @property
    def f1(self):
        """Return F1-measure."""
        return self.evaluation_results['f1']


class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    """Fallback definition of a File Splitting Model."""

    def __init__(self, *args, **kwargs):
        """Initialize the ContextAwareFileSplittingModel."""
        self.train_data = None
        self.test_data = None
        self.categories = None
        self.tokenizer = None
        self.first_page_spans = None
        sys.setrecursionlimit(99999999)

    def fit(self, *args, **kwargs) -> dict:
        """
        Gather the Spans unique for first Pages in a given stream of Documents.

        :return: Dictionary with unique first-page Span sets by Category ID.
        """
        first_page_spans = {}
        for category in self.categories:
            cur_first_page_spans = []
            cur_non_first_page_spans = []
            for doc in category.documents():
                doc = deepcopy(doc)
                doc.category = category
                doc = self.tokenizer.tokenize(doc)
                for page in doc.pages():
                    if page.number == 1:
                        cur_first_page_spans.append({span.offset_string for span in page.spans()})
                    else:
                        cur_non_first_page_spans.append({span.offset_string for span in page.spans()})
            if not cur_first_page_spans:
                cur_first_page_spans.append(set())
            true_first_page_spans = set.intersection(*cur_first_page_spans)
            if not cur_non_first_page_spans:
                cur_non_first_page_spans.append(set())
            true_not_first_page_spans = set.intersection(*cur_non_first_page_spans)
            true_first_page_spans = true_first_page_spans - true_not_first_page_spans
            first_page_spans[category.id_] = true_first_page_spans
        self.first_page_spans = first_page_spans
        return first_page_spans

    def save(self, model_path="", include_konfuzio=True):
        """
        Save the resulting set of first-page Spans by Category.

        :param model_path: Path to save the set to.
        :type model_path: str
        :param include_konfuzio: Enables pickle serialization as a value, not as a reference (for more info, read
        https://github.com/cloudpipe/cloudpickle#overriding-pickles-serialization-mechanism-for-importable-constructs).
        :type include_konfuzio: bool
        """
        if include_konfuzio:
            cloudpickle.register_pickle_by_value(konfuzio_sdk)
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
        temp_pkl_file_path = os.path.join(model_path, f'{get_timestamp()}_first_page_spans_tmp.cloudpickle')
        pkl_file_path = os.path.join(model_path, f'{get_timestamp()}_first_page_spans.pkl')
        logger.info('Saving model with cloudpickle')
        with open(temp_pkl_file_path, 'wb') as f:
            cloudpickle.dump(self.first_page_spans, f)
        logger.info('Compressing model with bz2')
        with open(temp_pkl_file_path, 'rb') as input_f:
            with bz2.open(pkl_file_path, 'wb') as output_f:
                shutil.copyfileobj(input_f, output_f)
        logger.info('Deleting cloudpickle file')
        os.remove(temp_pkl_file_path)
        return pkl_file_path

    def predict(self, page: Page) -> Page:
        """
        Take a Page as an input and return 1 for a first Page and 0 for a non-first Page.

        :param page: A Page to receive first or non-first label.
        :type page: Page
        :return: A Page with or without is_first_page label.
        """
        for category in self.categories:
            intersection = {span.offset_string for span in page.spans()}.intersection(
                self.first_page_spans[category.id_]
            )
            if len(intersection) > 0:
                page.is_first_page = True
        return page


class SplittingAI:
    """Split a given Document and return a list of resulting shorter Documents."""

    def __init__(self, model=""):
        """
        Initialize the class.

        :param model: A path to an existing .cloudpickle model or to a previously trained instance of
        ContextAwareFileSplittingModel().
        """
        self.tokenizer = ConnectedTextTokenizer()
        if model is str:
            self.model = ContextAwareFileSplittingModel()
            self.model.first_page_spans = load_model(model)
        else:
            self.model = model

    def _create_doc_from_page_interval(self, original_doc: Document, start_page: Page, end_page: Page) -> Document:
        pages_text = original_doc.text[start_page.start_offset : end_page.end_offset]
        new_doc = Document(project=LocalTextProject(), id_=None, text=pages_text)
        for page in original_doc.pages():
            if page.number in range(start_page.number, end_page.number):
                _ = Page(
                    id_=None,
                    original_size=(page.height, page.width),
                    document=new_doc,
                    start_offset=page.start_offset,
                    end_offset=page.end_offset,
                    number=page.number,
                )
        return new_doc

    def _suggest_first_pages(self, document: Document) -> Document:
        new_doc = self.tokenizer.tokenize(deepcopy(document))
        for page in new_doc.pages():
            self.model.predict(page)
        return new_doc

    def _suggest_page_split(self, document: Document) -> List[Document]:
        suggested_splits = []
        document = self.tokenizer.tokenize(deepcopy(document))
        for page in document.pages():
            if page.number == 1:
                suggested_splits.append(page)
            else:
                if hasattr(self.model.predict(page), 'is_first_page'):
                    suggested_splits.append(page)
        split_docs = []
        first_page = document.pages()[0]
        last_page = document.pages()[-1]
        for page_i, split_i in enumerate(suggested_splits):
            if page_i == 0:
                split_docs.append(self._create_doc_from_page_interval(document, first_page, split_i))
            elif page_i == len(split_docs):
                split_docs.append(self._create_doc_from_page_interval(document, split_i, last_page))
            else:
                split_docs.append(self._create_doc_from_page_interval(document, suggested_splits[page_i - 1], split_i))
        return split_docs

    def propose_split_documents(self, document: Document, return_pages: bool = False):
        """
        Propose a set of resulting documents from a single Documents.

        :param document: An input Document to be split.
        :type document: Document
        :param return_pages: A flag to enable returning a copy of an old Document with Pages marked .is_first_page on
        splitting points instead of a set of sub-Documents.
        :type return_pages: bool
        :return: A list of suggested new sub-Documents built from the original Document or a copy of an old Document
        with Pages marked .is_first_page on splitting points.
        """
        if return_pages:
            processed = self._suggest_first_pages(document)
        else:
            processed = self._suggest_page_split(document)
        return processed

    def evaluate_full(self, use_training_docs: bool = False) -> FileSplittingEvaluation:
        """
        Evaluate the filesplitting context-aware logic.

        :param use_training_docs: If enabled, runs evaluation on the training data to define its quality; if disabled,
        runs evaluation on the test data.
        :type use_training_docs: bool
        """
        pass
