"""Test Sentence Tokenizer."""
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
        cls.project = Project(id_=458)
        cls.tokenizer = SentenceTokenizer(mode='detectron')

        cls.document_1 = cls.project.get_document_by_id(615403)  # first 2 pages from YOLO9000 paper

    def test_sentence_document_1(self):
        """Test detectron Sentence tokenizer on 2 Page YOLO9000 Document."""
        virtual_doc = deepcopy(self.document_1)

        doc = self.tokenizer.tokenize(virtual_doc)

        assert len(doc.annotations(use_correct=False)) == 103

        pages = doc.pages()

        assert len(pages) == 2

        assert len(pages[0].annotations(use_correct=False)) == 37
        assert len(pages[1].annotations(use_correct=False)) == 66

    def test_sentence_document_1_use_detectron_labels(self):
        """Test detectron Sentence tokenizer on 2 Page YOLO9000 Document with create_detectron_labels option."""
        virtual_doc = deepcopy(self.document_1)

        tokenizer = SentenceTokenizer(mode='detectron', create_detectron_labels=True)

        virtual_doc = tokenizer(virtual_doc)

        assert len(virtual_doc.annotations(use_correct=False)) == 103
        assert len(virtual_doc.spans(use_correct=False)) == 225

        pages = virtual_doc.pages()

        assert len(pages) == 2

        assert len(pages[0].annotations(use_correct=False)) == 37
        assert len(pages[1].annotations(use_correct=False)) == 66

        assert virtual_doc.annotations(use_correct=False)[0].label.name == 'title'
        assert virtual_doc.annotations(use_correct=False)[1].label.name == 'text'
        assert virtual_doc.annotations(use_correct=False)[5].label.name == 'title'


class TestLineDistanceSentenceTokenizer(unittest.TestCase):
    """Test Line Distance Sentence Tokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=458)
        cls.tokenizer = SentenceTokenizer(mode='line_distance')

        cls.document_1 = cls.project.get_document_by_id(615403)  # first 2 pages from YOLO9000 paper

    def test_sentence_document_1(self):
        """Test detectron Sentence tokenizer on 2 Page YOLO9000 Document."""
        virtual_doc = deepcopy(self.document_1)

        doc = self.tokenizer.tokenize(virtual_doc)

        assert len(doc.annotations(use_correct=False)) == 97

        pages = doc.pages()

        assert len(pages) == 2

        assert len(pages[0].annotations(use_correct=False)) == 35
        assert len(pages[1].annotations(use_correct=False)) == 62


def test_no_bbox_document(caplog):
    """Test processing a Document with no Bboxes."""
    project = Project(id_=458)
    tokenizer = SentenceTokenizer(mode='detectron')
    document = project.get_document_by_id(615403)
    virtual_doc = deepcopy(document)
    virtual_doc.bboxes_available = False
    with caplog.at_level(logging.WARNING):
        tokenizer.tokenize(virtual_doc)
    assert 'Cannot tokenize Document' in caplog.text
