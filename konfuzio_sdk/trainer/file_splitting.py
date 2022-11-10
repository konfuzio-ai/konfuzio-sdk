"""Split a multi-Document file into a list of shorter documents."""
from typing import List

from konfuzio_sdk.data import Document, Page, Project


class SplittingAI:
    """Split a given Document and return a list of resulting shorter Documents."""

    def __init__(self, project_id=None):
        """
        Initialize the class.

        :param project_id: Project used for the intermediate document.
        :type project_id: int
        """
        self.project = Project(id_=project_id)

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

    def _suggest_page_split(self, document: Document) -> List[Document]:
        suggested_splits = []
        document.get_images()
        first_page_spans = set()
        another_page_spans = set()
        for page in document.pages():
            if page.number == 1:
                for span in page.spans():
                    first_page_spans.add(span)
            else:
                for span in page.spans():
                    another_page_spans.add(span)
        first_page_spans = first_page_spans - another_page_spans
        for page in document.pages():
            all_spans = len(page.spans())
            intersection_spans = 0
            for span in page.spans():
                if span in first_page_spans:
                    intersection_spans += 1
            first_page_percentage = intersection_spans / all_spans
            if first_page_percentage >= 0.5:
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
                split_docs.append(self._create_doc_from_page_interval(document, split_docs[page_i - 1], split_i))
        return split_docs

    def propose_split_documents(self, document: Document) -> List[Document]:
        """
        Propose a set of resulting documents from a single Documents.

        :param document: An input Document to be split.
        :return: A list of suggested new sub-Documents built from the original Document.
        """
        split_docs = self._suggest_page_split(document)
        return split_docs
