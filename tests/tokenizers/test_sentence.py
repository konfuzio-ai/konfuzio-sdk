"""Test sentence tokenizers."""
import logging
import unittest

from copy import deepcopy
from konfuzio_sdk.data import Project

from konfuzio_sdk.tokenizer.paragraph_and_sentence import SentenceTokenizer


logger = logging.getLogger(__name__)


class TestDetectronSentenceTokenizer(unittest.TestCase):
    """Test Detectron Sentence Tokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=458, update=True)
        cls.tokenizer = SentenceTokenizer(mode='detectron')

        cls.document_1 = cls.project.get_document_by_id(601418)  # Lorem ipsum test document
        cls.document_2 = cls.project.get_document_by_id(601419)  # Two column paper

    def test_sentence_document_1(self):
        """Test detectron Sentence tokenizer on Lorem ipsum Document."""
        virtual_doc = deepcopy(self.document_1)

        doc = self.tokenizer.tokenize(virtual_doc)

        assert len(doc.annotations(use_correct=False)) == 166

        pages = doc.pages()

        assert len(pages) == 3

        assert len(pages[0].annotations(use_correct=False)) == 51
        assert len(pages[1].annotations(use_correct=False)) == 58
        assert len(pages[2].annotations(use_correct=False)) == 57

    def test_sentence_document_2(self):
        """Test detectron Setence tokenizer on two column paper."""
        virtual_doc = deepcopy(self.document_2)

        doc = self.tokenizer.tokenize(virtual_doc)
        assert len(doc.annotations(use_correct=False)) == 403

        pages = doc.pages()
        assert len(pages) == 7

        assert len(pages[0].annotations(use_correct=False)) == 45
        assert len(pages[1].annotations(use_correct=False)) == 29
        assert len(pages[2].annotations(use_correct=False)) == 62
        assert len(pages[3].annotations(use_correct=False)) == 58
        assert len(pages[4].annotations(use_correct=False)) == 49
        assert len(pages[5].annotations(use_correct=False)) == 48
        assert len(pages[6].annotations(use_correct=False)) == 112


class TestLineDistanceParagraphTokenizer(unittest.TestCase):
    """Test Line Distance Sentence Tokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=458, update=True)
        cls.tokenizer = SentenceTokenizer(mode='line_distance')

        cls.document_1 = cls.project.get_document_by_id(601418)  # Lorem ipsum test document

    def test_paragraph_document_1(self):
        """Test detectron Sentence tokenizer on Lorem ipsum Document."""
        virtual_doc = deepcopy(self.document_1)

        doc = self.tokenizer.tokenize(virtual_doc)

        assert len(doc.annotations(use_correct=False)) == 160

        pages = doc.pages()

        assert len(pages) == 3

        assert len(pages[0].annotations(use_correct=False)) == 49
        assert len(pages[1].annotations(use_correct=False)) == 56
        assert len(pages[2].annotations(use_correct=False)) == 55
