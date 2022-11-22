"""Split a multi-Document file into a list of shorter documents."""
from abc import ABC
from copy import deepcopy
from typing import List, Set

from konfuzio_sdk.data import Document, Page, Project
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer


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

        :param category_id: Category by which Documents are filtered.
        :type category_id: int
        :return: A set of Spans unique to the first Pages.
        """
        first_page_spans = []
        not_first_page_spans = []
        for doc in self.project.documents:
            doc = deepcopy(doc)
            doc.category = self.project.get_category_by_id(self.category_id)
            doc = self.tokenizer.tokenize(deepcopy(doc))
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
            if len({span.offset_string for span in page.spans()}.intersection(first_page_spans)) > 1:
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
