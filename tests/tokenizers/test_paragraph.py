"""Test paragraph tokenizers."""
import logging
import unittest

from copy import deepcopy
from konfuzio_sdk.data import Project

from konfuzio_sdk.tokenizer.block import ParagraphTokenizer


logger = logging.getLogger(__name__)


class TestDetectronParagraphTokenizer(unittest.TestCase):
    """Test Detectron Paragraph Tokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=458, update=True)
        cls.tokenizer = ParagraphTokenizer(mode='detectron')

        cls.document_1 = cls.project.get_document_by_id(601418)  # Lorem ipsum test document
        cls.document_2 = cls.project.get_document_by_id(601419)  # Two column paper

    def test_paragraph_document_1(self):
        """Test detectron Paragraph tokenizer on Lorem ipsum Document."""
        virtual_doc = deepcopy(self.document_1)

        doc = self.tokenizer.tokenize(virtual_doc)

        assert len(doc.annotations(use_correct=False)) == 26

        pages = doc.pages()

        assert len(pages) == 3

        assert len(pages[0].annotations(use_correct=False)) == 8
        assert len(pages[1].annotations(use_correct=False)) == 10
        assert len(pages[2].annotations(use_correct=False)) == 8

    def test_paragraph_document_2(self):
        """Test detectron Paragraph tokenizer on two column paper."""
        virtual_doc = deepcopy(self.document_2)

        doc = self.tokenizer.tokenize(virtual_doc)
        assert len(doc.annotations(use_correct=False)) == 78

        pages = doc.pages()
        assert len(pages) == 7

        assert len(pages[0].annotations(use_correct=False)) == 16
        assert len(pages[1].annotations(use_correct=False)) == 10
        assert len(pages[2].annotations(use_correct=False)) == 12
        assert len(pages[3].annotations(use_correct=False)) == 14
        assert len(pages[4].annotations(use_correct=False)) == 12
        assert len(pages[5].annotations(use_correct=False)) == 10
        assert len(pages[6].annotations(use_correct=False)) == 4


class TestLineDistanceParagraphTokenizer(unittest.TestCase):
    """Test Line Distance Paragraph Tokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=458, update=True)
        cls.tokenizer = ParagraphTokenizer(mode='line_distance')

        cls.document_1 = cls.project.get_document_by_id(601418)  # Lorem ipsum test document

    def test_paragraph_document_1(self):
        """Test detectron Paragraph tokenizer on Lorem ipsum Document."""
        virtual_doc = deepcopy(self.document_1)

        doc = self.tokenizer.tokenize(virtual_doc)

        assert len(doc.annotations(use_correct=False)) == 29

        pages = doc.pages()

        assert len(pages) == 3

        assert len(pages[0].annotations(use_correct=False)) == 8
        assert len(pages[1].annotations(use_correct=False)) == 12
        assert len(pages[2].annotations(use_correct=False)) == 9
