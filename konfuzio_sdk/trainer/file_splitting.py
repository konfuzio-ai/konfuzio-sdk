"""Find similarities between Documents or Pages via comparison between their texts."""
import abc
import json
import logging
import os
import pathlib
import sys

from copy import deepcopy
from pympler import asizeof
from typing import List, Union

from konfuzio_sdk.data import Document, Page
from konfuzio_sdk.evaluate import FileSplittingEvaluation
from konfuzio_sdk.trainer.information_extraction import load_model, BaseModel
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.utils import get_timestamp, normalize_memory

logger = logging.getLogger(__name__)


class AbstractFileSplittingModel(BaseModel, metaclass=abc.ABCMeta):
    """Abstract class for the filesplitting model."""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the class."""
        self.model_type = 'file_splitting'
        self.output_dir = None

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the custom model on the training Documents."""

    @abc.abstractmethod
    def predict(self, page: Page) -> Page:
        """
        Take a Page as an input and return 1 for first page and 0 for not first page.

        :param page: A Page to label first or non-first.
        :type page: Page
        :return: A Page with or without is_first_page label.
        """

    def reduce_model_weight(self, max_ram, *args, **kwargs):
        """Remove all non-strictly necessary parameters before saving."""
        self.lose_weight()

        # if no argument passed, get project max_ram
        max_ram = self.documents[0].project.max_ram

        max_ram = normalize_memory(max_ram)
        sys.setrecursionlimit(99999999)  # ?

        if max_ram and asizeof.asizeof(self) > max_ram:
            raise MemoryError(f"AI model memory use ({asizeof.asizeof(self)}) exceeds maximum ({max_ram=}).")

    def generate_pickle_output_paths(self, *args, **kwargs):
        """Generate paths for temporary and resulting pickle files."""
        temp_pkl_file_path = os.path.join(
            self.output_dir, f'{get_timestamp()}_{self.name_lower()}_{self.documents[0].project.id_}_tmp.pkl'
        )
        pkl_file_path = os.path.join(
            self.output_dir, f'{get_timestamp()}_{self.name_lower()}_{self.documents[0].project.id_}.pkl'
        )
        return temp_pkl_file_path, pkl_file_path


class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    """Fallback definition of a File Splitting Model."""

    def __init__(self, *args, **kwargs):
        """Initialize the ContextAwareFileSplittingModel."""
        super().__init__()
        self.documents = None
        self.test_documents = None
        self.categories = None
        self.tokenizer = None
        self.first_page_spans = None
        self.path = None
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
            first_page_spans[category.id_] = list(true_first_page_spans)
        self.first_page_spans = first_page_spans
        return first_page_spans

    def save(self, save_json=True, include_konfuzio=False, max_ram=None, reduce_weight: bool = False) -> str:
        """
        Save the resulting set of first-page Spans by Category.

        :param save_json: Whether to save JSON of first_page_spans or a pickle of the whole class.
        :type save_json: bool
        :param include_konfuzio: Enables pickle serialization as a value, not as a reference (for more info, read
        https://github.com/cloudpipe/cloudpickle#overriding-pickles-serialization-mechanism-for-importable-constructs).
        :type include_konfuzio: bool
        :param max_ram: Specify maximum memory usage condition to save model.
        :raises MemoryError: When the size of the model in memory is greater than the maximum value.
        :param reduce_weight: Remove all non-strictly necessary parameters before saving.
        :type reduce_weight: bool
        """
        if save_json:
            self.path = self.output_dir + f'/{get_timestamp()}_first_page_spans.json'
            pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            with open(self.path, 'w+') as f:
                json.dump(self.first_page_spans, f)
        else:
            self.path = super().save(include_konfuzio=include_konfuzio, max_ram=max_ram, reduce_weight=reduce_weight)
        return self.path

    def load_json(self, model_path=""):
        """
        Load JSON with previously gathered first_page_spans.

        :param model_path: Path for the JSON.
        :type model_path: str
        """
        with open(model_path, 'r') as f:
            spans = json.load(f)
        # converting str category.id_ values to back int because JSON converts them to str
        spans = {int(k): v for k, v in spans.items()}
        self.first_page_spans = spans

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
                break
        return page

    def lose_weight(self):
        """Remove unnecessary data to reduce size of the model."""
        self.documents = None
        self.test_documents = None


class SplittingAI:
    """Split a given Document and return a list of resulting shorter Documents."""

    def __init__(self, model=""):
        """
        Initialize the class.

        :param model: A path to an existing .cloudpickle model or to a previously trained instance of
        ContextAwareFileSplittingModel().
        """
        self.tokenizer = ConnectedTextTokenizer()
        if isinstance(model, str):
            self.model = ContextAwareFileSplittingModel()
            self.model.first_page_spans = load_model(model)
        else:
            self.model = model

    def _create_doc_from_page_interval(self, original_doc: Document, start_page: Page, end_page: Page) -> Document:
        pages_text = original_doc.text[start_page.start_offset : end_page.end_offset]
        new_doc = Document(project=original_doc.project, id_=None, text=pages_text)
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

    def _suggest_first_pages(self, document: Document, inplace: bool = False) -> Document:
        if inplace:
            new_doc = self.tokenizer.tokenize(document)
        else:
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
                if self.model.predict(page).is_first_page:
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

    def propose_split_documents(
        self, document: Document, return_pages: bool = False, inplace: bool = False
    ) -> Union[Document, List]:
        """
        Propose a set of resulting documents from a single Documents.

        :param document: An input Document to be split.
        :type document: Document
        :param inplace: Whether changes are applied to an initially passed Document, changing it, or to its deepcopy.
        :type inplace: bool
        :param return_pages: A flag to enable returning a copy of an old Document with Pages marked .is_first_page on
        splitting points instead of a set of sub-Documents.
        :type return_pages: bool
        :return: A list of suggested new sub-Documents built from the original Document or a copy of an old Document
        with Pages marked .is_first_page on splitting points.
        """
        if return_pages:
            processed = self._suggest_first_pages(document, inplace)
        else:
            processed = self._suggest_page_split(document)
        return processed

    def evaluate_full(self, use_training_docs: bool = False) -> FileSplittingEvaluation:
        """
        Evaluate the SplittingAI's performance.

        :param use_training_docs: If enabled, runs evaluation on the training data to define its quality; if disabled,
        runs evaluation on the test data.
        :type use_training_docs: bool
        :return: Evaluation information for the filesplitting context-aware logic.
        """
        evaluation_list = []
        if not use_training_docs:
            evaluation_docs = self.model.test_documents
        else:
            evaluation_docs = self.model.documents
        for doc in evaluation_docs:
            pred = self.propose_split_documents(doc, return_pages=True)
            evaluation_list.append((doc, pred))
        self.full_evaluation = FileSplittingEvaluation(evaluation_list)
        return self.full_evaluation
