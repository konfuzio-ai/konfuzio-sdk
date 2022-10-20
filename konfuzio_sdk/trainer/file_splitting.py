"""Split a multi-Document file into a list of shorter documents based on model's prediction."""

import pickle

from nltk import word_tokenize
from typing import List

from konfuzio_sdk.data import Document, Page, Project


class FusionModel:
    """Train a fusion model for correct splitting of multi-file documents."""

    def __init__(self, project_id: int, split_point: float = 0.5):
        """Initialize project, training and testing data, and a split point for training dataset."""
        self.project = Project(id_=project_id)
        self.train_data = self.project.documents
        self.test_data = self.project.test_documents
        split_point = int(split_point * len(self.train_data))


class PageSplitting:
    """Split a given document and return a list of resulting shorter documents."""

    def __init__(self, model_path: str):
        """Load model, tokenizer, vocabulary and categorization_pipeline."""
        self.load(model_path)

    def save(self) -> None:
        """Save model, tokenizer, and vocabulary used for document splitting."""
        pickle.dump((self.model, self.tokenizer, self.vocab))

    def load(self, path: str):
        """Load model, tokenizer, and vocabulary from a previously pickled file."""
        self.model, self.tokenizer, self.vocab = pickle.load(open(path))

    def _predict(self, page_text: str) -> bool:
        tokens = word_tokenize(page_text)
        tokens = [t for t in tokens if t in self.vocab]
        line = ' '.join(tokens)
        encoded = self.tokenizer.texts_to_matrix([line], mode='freq')
        predicted = self.model.predict(encoded, verbose=0)
        return bool(round(predicted[0, 0]))

    def _create_doc_from_page_interval(self, original_doc: Document, start_page: Page, end_page: Page) -> Document:
        pages_text = original_doc.text[start_page.start_offset : end_page.end_offset]
        new_doc = Document(id_=None, text=pages_text)
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
        for page_i, page in enumerate(document.pages()):
            is_first_page = self._predict(page.text)
            if is_first_page:
                suggested_splits.append(page_i)

        split_docs = []
        last_page = [
            page for page in document.pages() if page.number == max([page.number for page in document.pages()])
        ][0]
        for page_i, split_i in enumerate(suggested_splits):
            if page_i == 0:
                split_docs.append(self._create_doc_from_page_interval(document, page_i, split_i))
            elif page_i == len(split_docs):
                split_docs.append(self._create_doc_from_page_interval(document, split_i, last_page))
            else:
                split_docs.append(self._create_doc_from_page_interval(document, split_docs[page_i - 1], split_i))
        return split_docs

    def propose_mappings(self, document: Document) -> List[Document]:
        """
        Propose a set of resulting documents from a single Documents.

        :param document: An input Document to be split.
        """
        split_docs = self._suggest_page_split(document)
        return split_docs
