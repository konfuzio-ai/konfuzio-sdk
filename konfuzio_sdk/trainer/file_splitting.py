"""Find similarities between Documents or Pages via comparison between their texts."""
import abc
import logging
import os
import sys

from copy import deepcopy
from pympler import asizeof
from typing import List

from konfuzio_sdk.data import Document, Page, Category
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
        super().__init__()
        self.output_dir = None

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the custom model on the training Documents."""  # there is no return

    @abc.abstractmethod
    def predict(self, page: Page) -> Page:
        """
        Take a Page as an input and reassign is_first_page attribute's value if necessary.

        :param page: A Page to label first or non-first.
        :type page: Page
        :return: Page.
        """

    @property
    def temp_pkl_file_path(self) -> str:
        """Generate a path for temporary pickle file."""
        temp_pkl_file_path = os.path.join(
            self.output_dir,
            f'{get_timestamp(konfuzio_format="%Y-%m-%d-%H-%M")}_{self.name_lower()}_{self.project.id_}_' f'tmp.pkl',
        )
        return temp_pkl_file_path

    @property
    def pkl_file_path(self) -> str:
        """Generate a path for a resulting pickle file."""
        pkl_file_path = os.path.join(
            self.output_dir,
            f'{get_timestamp(konfuzio_format="%Y-%m-%d-%H-%M")}_{self.name_lower()}_{self.project.id_}' f'.pkl',
        )
        return pkl_file_path

    def lose_weight(self):
        """Remove all data not necessary for prediction."""
        self.documents = None
        self.test_documents = None

    def reduce_model_weight(self):
        """Remove all non-strictly necessary parameters before saving."""
        self.lose_weight()
        self.tokenizer.lose_weight()

    def ensure_model_memory_usage_within_limit(self, max_ram):
        """
        Ensure that a model is not exceeding allowed max_ram.

        :param max_ram: Specify maximum memory usage condition to save model.
        """
        if not max_ram:
            max_ram = self.documents[0].project.max_ram

        max_ram = normalize_memory(max_ram)

        if max_ram and asizeof.asizeof(self) > max_ram:
            raise MemoryError(f"AI model memory use ({asizeof.asizeof(self)}) exceeds maximum ({max_ram=}).")

        sys.setrecursionlimit(99999999)

    def restore_category_documents_for_eval(self):
        """Restore Documents deleted when reducing weight in case there's evaluation needed."""
        self.documents = [document for category in self.categories for document in category.documents()]
        self.test_documents = [document for category in self.categories for document in category.test_documents()]


class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    """Fallback definition of a File Splitting Model."""

    def __init__(self, categories: List[Category], tokenizer, *args, **kwargs):
        """
        Initialize the ContextAwareFileSplittingModel.

        :param categories: A list of Categories to run training/prediction of the model on.
        :type categories: List[Category]
        :param tokenizer: Tokenizer used for processing Documents on fitting when searching for exclusive first-page
        strings.
        :raises ValueError: When an empty list of Categories is passed into categories argument.
        :raises ValueError: When a list passed into categories contains elements other than Categories.
        :raises ValueError: When a list passed into categories contains at least one Category with no documents or test
        documents.
        :raises ValueError: When a list passed into categories contains Categories from different Projects.
        """
        super().__init__()
        self.name = self.__class__.__name__
        if not len(categories):
            raise ValueError("Cannot initialize ContextAwareFileSplittingModel on an empty list.")
        for category in categories:
            if not type(category) == Category:
                raise ValueError("All elements of the list have to be Categories.")
            if not len(category.documents()):
                raise ValueError(f'{category} does not have Documents and cannot be used for training.')
            if not len(category.test_documents()):
                raise ValueError(f'{category} does not have test Documents.')
        projects = set([category.project for category in categories])
        if len(projects) > 1:
            raise ValueError("All Categories have to belong to the same Project.")
        self.categories = categories
        self.project = self.categories[0].project  # we ensured that at least one Category is present
        self.output_dir = self.project.model_folder
        self.documents = [document for category in self.categories for document in category.documents()]
        self.test_documents = [document for category in self.categories for document in category.test_documents()]
        self.tokenizer = tokenizer
        self.path = None
        self._used_tokenizer = None

    def fit(self, allow_empty_categories: bool = False, *args, **kwargs):
        """
        Gather the strings exclusive for first Pages in a given stream of Documents.

        Exclusive means that each of these strings appear only on first Pages of Documents within a Category.

        :param allow_empty_categories: To allow returning empty list for a Category if no exclusive first-page strings
        were found during fitting (which means prediction would be impossible for a Category).
        :type allow_empty_categories: bool
        :raises ValueError: When allow_empty_categories is False and no exclusive first-page strings were found for
        at least one Category.
        """
        if not self._used_tokenizer:
            self._used_tokenizer = self.tokenizer
        else:
            if self.tokenizer != self._used_tokenizer:
                logger.warning(
                    "Assigned tokenizer does not correspond to the one previously used within this instance."
                    "All previously found exclusive first-page strings within each Category will be removed "
                    "and replaced with the newly generated ones."
                )
                for category in self.categories:
                    category._exclusive_first_page_strings = None
        for category in self.categories:
            if not category.exclusive_first_page_strings:
                category.collect_exclusive_first_page_strings(self.tokenizer)
                if allow_empty_categories:
                    logger.warning(
                        f'No exclusive first-page strings were found for {category}, so it will not be used '
                        f'at prediction.'
                    )
                else:
                    raise ValueError(f'No exclusive first-page strings were found for {category}.')

    def predict(self, page: Page) -> Page:
        """
        Take a Page as an input and predict it as first or non-first.

        :param page: A Page to receive first or non-first label.
        :type page: Page
        :raises ValueError: When at least one Category does not have exclusive_first_page_strings.
        :return: A Page with a newly predicted is_first_page attribute.
        """
        for category in self.categories:
            if not category.exclusive_first_page_strings:
                raise ValueError(f"Cannot run prediction as {category} does not have exclusive_first_page_strings.")
        page.is_first_page = False
        for category in self.categories:
            intersection = {span.offset_string for span in page.spans()}.intersection(
                category.exclusive_first_page_strings
            )
            if len(intersection) > 0:
                page.is_first_page = True
                break
        return page


class SplittingAI:
    """Split a given Document and return a list of resulting shorter Documents."""

    def __init__(self, model):
        """
        Initialize the class.

        :param model: A path to an existing .cloudpickle model or to a previously trained instance of
        ContextAwareFileSplittingModel().
        """
        self.tokenizer = ConnectedTextTokenizer()
        if isinstance(model, str):
            self.model = load_model(model)
        else:
            self.model = model
        if not issubclass(type(self.model), AbstractFileSplittingModel):
            raise ValueError("The model is not inheriting from AbstractFileSplittingModel class.")

    def _suggest_first_pages(self, document: Document, inplace: bool = False) -> List[Document]:
        """
        Run prediction on Document's Pages, predicting them to be first or non-first.

        :param document: A Document to run prediction on.
        :type document: Document
        :param inplace: To create a copy of a Document or to run prediction over the original one.
        :type inplace: bool
        :returns: A list with a single Document – for the sake of unity across different prediction methods' returns
        in the class.
        """
        if inplace:
            new_doc = self.tokenizer.tokenize(document)
        else:
            new_doc = self.tokenizer.tokenize(deepcopy(document))
        for page in new_doc.pages():
            self.model.predict(page)
        return [new_doc]

    def _suggest_page_split(self, document: Document) -> List[Document]:
        """
        Create a list of sub-Documents built from the original Document, split.

        :param document: A Document to run prediction on.
        :type document: Document
        :returns: A list of sub-Documents built from the original Document, based on model's prediction of first Pages.
        """
        suggested_splits = []
        document = self.tokenizer.tokenize(deepcopy(document))
        for page in document.pages():
            if page.number == 1:
                suggested_splits.append(page)
            else:
                if self.model.predict(page).is_first_page:
                    suggested_splits.append(page)
        if len(suggested_splits) == 1:
            return [document]
        else:
            split_docs = []
            first_page = document.pages()[0]
            last_page = document.pages()[-1]
            for page_i, split_i in enumerate(suggested_splits):
                if page_i == 0:
                    split_docs.append(document.create_subdocument_from_page_range(first_page, split_i))
                elif page_i == len(split_docs):
                    split_docs.append(document.create_subdocument_from_page_range(split_i, last_page))
                else:
                    split_docs.append(
                        document.create_subdocument_from_page_range(suggested_splits[page_i - 1], split_i)
                    )
        return split_docs

    def propose_split_documents(
        self, document: Document, return_pages: bool = False, inplace: bool = False
    ) -> List[Document]:
        """
        Propose a set of resulting documents from a single Documents.

        :param document: An input Document to be split.
        :type document: Document
        :param inplace: Whether changes are applied to an initially passed Document, changing it, or to its deepcopy.
        :type inplace: bool
        :param return_pages: A flag to enable returning a copy of an old Document with Pages marked .is_first_page on
        splitting points instead of a set of sub-Documents.
        :type return_pages: bool
        :return: A list of suggested new sub-Documents built from the original Document or a list with a Document
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
        pred_docs = []
        if not use_training_docs:
            original_docs = self.model.test_documents
        else:
            original_docs = self.model.documents
        for doc in original_docs:
            predictions = self.propose_split_documents(doc, return_pages=True)
            assert len(predictions) == 1
            pred_docs.append(predictions[0])
        self.full_evaluation = FileSplittingEvaluation(original_docs, pred_docs)
        return self.full_evaluation
