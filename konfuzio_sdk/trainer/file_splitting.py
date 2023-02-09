"""
Process Documents that consist of several files and propose splitting them into the Sub-Documents accordingly.

We suggest a context-aware approach based on scanning Category's Documents and finding strings exclusive for first
Pages of all Documents within the Category. Upon predicting whether a Page is a potential splitting point (meaning
whether it is first or not), we compare Page's contents to these exclusive first-page strings; if there is occurrence of
at least one such string, we mark a Page to be first (thus meaning it is a splitting point).

A ContextAwareFileSplittingModel instance can be used with an interface provided by SplittingAI – this class accepts a
whole Document instead of a single Page and proposes splitting points or splits the original Documents.

For developing a custom file-splitting approach, we propose an abstract class.
"""
import abc
import logging
import os

from copy import deepcopy
from typing import List

from konfuzio_sdk.data import Document, Page, Category
from konfuzio_sdk.evaluate import FileSplittingEvaluation
from konfuzio_sdk.trainer.information_extraction import load_model, BaseModel
from konfuzio_sdk.utils import get_timestamp

logger = logging.getLogger(__name__)


class AbstractFileSplittingModel(BaseModel, metaclass=abc.ABCMeta):
    """Abstract class for the filesplitting model."""

    @abc.abstractmethod
    def __init__(self, categories: List[Category], *args, **kwargs):
        """
        Initialize the class.

        :param categories: A list of Categories to run training/prediction of the model on.
        :type categories: List[Category]
        """
        super().__init__()
        self.output_dir = None
        if not categories:
            raise ValueError("Cannot initialize ContextAwareFileSplittingModel on an empty list.")
        for category in categories:
            if not isinstance(category, Category):
                raise ValueError("All elements of the list have to be Categories.")
            if not category.documents():
                raise ValueError(f'{category} does not have Documents and cannot be used for training.')
            if not category.test_documents():
                raise ValueError(f'{category} does not have test Documents.')
        projects = set([category.project for category in categories])
        if len(projects) > 1:
            raise ValueError("All Categories have to belong to the same Project.")
        self.categories = categories
        self.project = self.categories[0].project  # we ensured that at least one Category is present
        self.documents = [document for category in self.categories for document in category.documents()]
        self.test_documents = [document for category in self.categories for document in category.test_documents()]
        self.tokenizer = None
        self.requires_text = False
        self.requires_images = False

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
            f'{get_timestamp()}_{self.name_lower()}_{self.project.id_}_' f'tmp.pkl',
        )
        return temp_pkl_file_path

    @property
    def pkl_file_path(self) -> str:
        """Generate a path for a resulting pickle file."""
        pkl_file_path = os.path.join(
            self.output_dir,
            f'{get_timestamp()}_{self.name_lower()}_{self.project.id_}' f'.pkl',
        )
        return pkl_file_path


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
        super().__init__(categories=categories)
        self.output_dir = self.project.model_folder
        self.tokenizer = tokenizer
        self.requires_text = True
        self.requires_images = False

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
        for category in self.categories:
            # method exclusive_first_page_strings fetches a set of first-page strings exclusive among the Documents
            # of a given Category. they can be found in _exclusive_first_page_strings attribute of a Category after
            # the method has been run. this is needed so that the information remains even if local variable
            # cur_first_page_strings is lost.
            cur_first_page_strings = category.exclusive_first_page_strings(tokenizer=self.tokenizer)
            if not cur_first_page_strings:
                if allow_empty_categories:
                    logger.warning(
                        f'No exclusive first-page strings were found for {category}, so it will not be used '
                        f'at prediction.'
                    )
                else:
                    raise ValueError(f'No exclusive first-page strings were found for {category}.')

    def predict(self, page: Page) -> Page:
        """
        Predict a Page as first or non-first.

        :param page: A Page to receive first or non-first label.
        :type page: Page
        :raises ValueError: When at least one Category does not have exclusive_first_page_strings.
        :return: A Page with a newly predicted is_first_page attribute.
        """
        self.check_is_ready()
        page.is_first_page = False
        for category in self.categories:
            cur_first_page_strings = category.exclusive_first_page_strings(tokenizer=self.tokenizer)
            intersection = {span.offset_string.strip('\f').strip('\n') for span in page.spans()}.intersection(
                cur_first_page_strings
            )
            if len(intersection) > 0:
                page.is_first_page = True
                break
        return page

    def check_is_ready(self):
        """Check file splitting model is ready for inference."""
        if self.tokenizer is None:
            raise AttributeError(f'{self} missing Tokenizer.')

        if not self.categories:
            raise AttributeError(f'{self} requires Categories.')

        for category in self.categories:
            if not category.exclusive_first_page_strings(tokenizer=self.tokenizer):
                # exclusive_first_page_strings calls an implicit _exclusive_first_page_strings attribute once it was
                # already calculated during fit() method so it is not a recurrent calculation each time.
                raise ValueError(f"Cannot run prediction as {category} does not have _exclusive_first_page_strings.")


class SplittingAI:
    """Split a given Document and return a list of resulting shorter Documents."""

    def __init__(self, model):
        """
        Initialize the class.

        :param model: A path to an existing .cloudpickle model or to a previously trained instance of
        ContextAwareFileSplittingModel().
        :raises ValueError: When the model is not inheriting from AbstractFileSplittingModel class.
        """
        self.model = load_model(model) if isinstance(model, str) else model
        if not issubclass(type(self.model), AbstractFileSplittingModel):
            raise ValueError("The model is not inheriting from AbstractFileSplittingModel class.")
        self.tokenizer = self.model.tokenizer

    def _suggest_first_pages(self, document: Document, inplace: bool = False) -> List[Document]:
        """
        Run prediction on Document's Pages, marking them as first or non-first.

        :param document: The Document to predict the Pages of.
        :type document: Document
        :param inplace: Whether to predict the Pages on the original Document or on a copy.
        :type inplace: bool
        :returns: A list containing the modified Document.
        """
        if not self.model.requires_images:
            if inplace:
                document = self.tokenizer.tokenize(document)
            else:
                document = self.tokenizer.tokenize(deepcopy(document))
            for page in document.pages():
                self.model.predict(page)
        else:
            for page in document.pages():
                self.model.predict(page)
        return [document]

    def _suggest_page_split(self, document: Document) -> List[Document]:
        """
        Create a list of Sub-Documents built from the original Document, split.

        :param document: The document to suggest Page splits for.
        :type document: Document
        :returns: A list of Sub-Documents created from the original Document, split at predicted first Pages.
        """
        suggested_splits = []
        if self.tokenizer:
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
                    split_docs.append(
                        document.create_subdocument_from_page_range(
                            first_page, suggested_splits[page_i + 1], include=False
                        )
                    )
                elif page_i == len(suggested_splits) - 1:
                    split_docs.append(document.create_subdocument_from_page_range(split_i, last_page, include=True))
                else:
                    split_docs.append(
                        document.create_subdocument_from_page_range(
                            split_i, suggested_splits[page_i + 1], include=False
                        )
                    )
        return split_docs

    def propose_split_documents(
        self, document: Document, return_pages: bool = False, inplace: bool = False
    ) -> List[Document]:
        """
        Propose a set of resulting documents from a single Documents.

        :param document: An input Document to be split.
        :type document: Document
        :param inplace: Whether changes are applied to the input Document, changing it, or to a deepcopy of it.
        :type inplace: bool
        :param return_pages: A flag to enable returning a copy of an old Document with Pages marked .is_first_page on
        splitting points instead of a set of Sub-Documents.
        :type return_pages: bool
        :return: A list of suggested new Sub-Documents built from the original Document or a list with a Document
        with Pages marked .is_first_page on splitting points.
        """
        if self.model.requires_images:
            document.get_images()
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
