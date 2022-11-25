"""Find similarities between Documents or Pages via comparison between their texts."""
import abc
import pickle

from abc import ABC
from copy import deepcopy
from typing import List, Set

from konfuzio_sdk.data import Document, Page, Project
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer


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

    def calculate_metrics(self, use_training_docs: bool = False):
        """
        Calculate precision, recall, and F1 measure for the custom model.

        :param use_training_docs: A flag for using training Documents for evaluation or not.
        :type use_training_docs: bool
        :return: Calculated precision, recall, and F1 measure.
        """
        true_positive = 0
        false_positive = 0
        false_negative = 0
        if use_training_docs:
            list_of_pages = [
                page for category in self.categories for document in category.documents() for page in document.pages()
            ]
        else:
            list_of_pages = [
                page
                for category in self.categories
                for document in category.test_documents()
                for page in document.pages()
            ]
        for page in list_of_pages:
            pred = self.predict(page)
            if page.number == 1 and pred == 1:
                true_positive += 1
            elif page.number == 1 and pred == 0:
                false_negative += 1
            elif page.number == 0 and pred == 1:
                false_positive += 1
        if true_positive + false_positive != 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0
        if true_positive + false_negative != 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        return precision, recall, f1

    @abc.abstractmethod
    def predict(self, page: Page) -> int:
        """
        Take a Page as an input and return 1 for first page and 0 for not first page.

        :param page: A Page to label first or non-first.
        :type page: Page
        :return: A label of a first or a non-first Page.
        """


class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    """Fallback definition of a File Splitting Model."""

    def __init__(self, *args, **kwargs):
        """Initialize the ContextAwareFileSplittingModel."""
        self.train_data = None
        self.test_data = None
        self.categories = None
        self.tokenizer = None
        self.first_page_spans = None

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
                doc = self.tokenizer.tokenize(deepcopy(doc))
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

    def save(self, model_path=""):
        """
        Save the resulting set of first-page Spans by Category.

        :param model_path: Path to save the set to.
        :type model_path: str
        """
        with open(model_path + '/first_page_spans.pickle', "wb") as pickler:
            pickle.dump(self.first_page_spans, pickler)

    def predict(self, page: Page) -> int:
        """
        Take a Page as an input and return 1 for a first Page and 0 for a non-first Page.

        :param page: A Page to receive first or non-first label.
        :type page: Page:
        :return: A label of a first or a non-first Page.
        """
        intersection_lengths = {}
        for category in self.categories:
            intersection = len(
                {span.offset_string for span in page.spans()}.intersection(self.first_page_spans[category.id_])
            )
            if intersection > 0:
                intersection_lengths[category.id_] = len(
                    {span.offset_string for span in page.spans()}.intersection(self.first_page_spans[category.id_])
                )
        if len(intersection_lengths) > 0:
            return 1
        else:
            return 0


def find_common_spans_document(documents, category, tokenizer) -> Set:
    """
    Gather the Spans common for a group of Documents.

    :param documents: A group of Documents to search unique features of.
    :type documents: list
    :param category: A Category for which the search is done.
    :type category: Category
    :param tokenizer: A tokenizer to split Documents into Spans.
    :return: A set of Spans typical for this Category.
    """
    span_sets = []
    for doc in documents:
        doc = deepcopy(doc)
        doc.category = category
        doc = tokenizer.tokenize(doc)
        span_sets.append({span.offset_string for span in doc.spans()})
    common_spans = set.intersection(*span_sets)
    return common_spans


def find_unique_spans_page(documents, category, tokenizer, **kwargs) -> Set:
    # the name for the function can be rethinked
    """
    Gather the Spans unique in a defined way for a stream of Pages.

    :param documents: A group of Documents in Pages of which to search for the common Spans.
    :type documents: list
    :param category: A Category for which the search is done.
    :type category: Category
    :param tokenizer: A tokenizer to split Documents into Spans.
    :return: A set of unique Spans.
    """
    # we can define possible modes of usage and suggest using them for suitable occurences
    if kwargs['mode'] == 'file_splitting':
        first_page_spans = []
        not_first_page_spans = []
        for doc in documents:
            doc = deepcopy(doc)
            doc.category = category
            doc = tokenizer.tokenize(deepcopy(doc))
            for page in doc.pages():
                if page.number == 1:
                    first_page_spans.append({span.offset_string for span in page.spans()})
                else:
                    not_first_page_spans.append({span.offset_string for span in page.spans()})
        if not first_page_spans:
            first_page_spans.append(set())
        true_first_page_spans = set.intersection(*first_page_spans)
        if not not_first_page_spans:
            not_first_page_spans.append(set())
        true_not_first_page_spans = set.intersection(*not_first_page_spans)
        true_first_page_spans = true_first_page_spans - true_not_first_page_spans
    return true_first_page_spans


class BaseCommonFeatureSearcher(ABC):
    """Create a class to set constraints for the derivatives."""

    def __init__(self):
        """Initialize the tokenizer."""
        self.tokenizer = ConnectedTextTokenizer()


class SplittingAI(BaseCommonFeatureSearcher):
    """Split a given Document and return a list of resulting shorter Documents."""

    def __init__(self, project_id=None, category_id=None):
        """
        Initialize the class.

        :param project_id: Project used for the intermediate document.
        :type project_id: int
        """
        super().__init__()
        self.project = Project(id_=project_id)
        self.train_data = self.project.documents
        self.category_id = category_id

    def _create_doc_from_page_interval(self, original_doc: Document, start_page: Page, end_page: Page) -> Document:
        pages_text = original_doc.text[start_page.start_offset : end_page.end_offset]
        new_doc = Document(project=self.project, id_=None, text=pages_text)
        for page in original_doc.pages():
            if page.number in range(start_page.number, end_page.number):
                _ = Page(
                    document=new_doc,
                    start_offset=page.start_offset,
                    end_offset=page.end_offset,
                    page_number=page.number,
                )
        return new_doc

    def train(self) -> Set:
        """
        Gather the Spans unique for the first Pages.

        :return: A set of Spans unique to the first Pages.
        """
        first_page_spans = []
        not_first_page_spans = []
        for doc in self.project.documents:
            doc = deepcopy(doc)
            doc.category = self.project.get_category_by_id(self.category_id)
            doc = self.tokenizer.tokenize(doc)
            for page in doc.pages():
                if page.number == 1:
                    first_page_spans.append({span.offset_string for span in page.spans()})
                else:
                    not_first_page_spans.append({span.offset_string for span in page.spans()})
        if not first_page_spans:
            first_page_spans.append(set())
        true_first_page_spans = set.intersection(*first_page_spans)
        if not not_first_page_spans:
            not_first_page_spans.append(set())
        true_not_first_page_spans = set.intersection(*not_first_page_spans)
        true_first_page_spans = true_first_page_spans - true_not_first_page_spans
        return true_first_page_spans

    def _suggest_page_split(self, document: Document, first_page_spans: Set) -> List[Document]:
        suggested_splits = []
        for page in document.pages():
            if len({span.offset_string for span in page.spans()}.intersection(first_page_spans)) > 0:
                suggested_splits.append(page)
        split_docs = []
        first_page = document.pages()[0]
        last_page = document.pages()[-1]
        if not suggested_splits:
            return [document]
        for page_i, split_i in enumerate(suggested_splits):
            if page_i == 0:
                split_docs.append(self._create_doc_from_page_interval(document, first_page, split_i))
            elif page_i == len(split_docs):
                split_docs.append(self._create_doc_from_page_interval(document, split_i, last_page))
            else:
                split_docs.append(self._create_doc_from_page_interval(document, suggested_splits[page_i - 1], split_i))
        return split_docs

    def propose_split_documents(self, document: Document, first_page_spans: Set) -> List[Document]:
        """
        Propose a set of resulting documents from a single Documents.

        :param document: An input Document to be split.
        :type document: Document
        :param first_page_spans: A set of Spans unique for the first Pages in the training data.
        :type first_page_spans: set
        :return: A list of suggested new sub-Documents built from the original Document.
        """
        split_docs = self._suggest_page_split(document, first_page_spans)
        return split_docs
