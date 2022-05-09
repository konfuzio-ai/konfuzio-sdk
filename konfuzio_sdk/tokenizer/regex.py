"""Regex tokenizers."""
import logging
import time

import pandas as pd

from konfuzio_sdk.data import Annotation, Document, Category, Span
from konfuzio_sdk.evaluate import compare
from konfuzio_sdk.regex import regex_matches
from konfuzio_sdk.tokenizer.base import AbstractTokenizer, ListTokenizer, ProcessingStep
from konfuzio_sdk.utils import sdk_isinstance

logger = logging.getLogger(__name__)


class RegexTokenizer(AbstractTokenizer):
    """Tokenizer based on a single regex."""

    def __init__(self, regex: str):
        """Initialize the RegexTokenizer."""
        self.regex = regex
        self.processing_steps = []

    def fit(self, category: Category):
        """Fit the tokenizer accordingly with the Documents of the Category."""
        assert sdk_isinstance(category, Category)
        return self

    def tokenize(self, document: Document) -> Document:
        """
        Create Annotations with 1 Span based on the result of the Tokenizer.

        :param document: Document to tokenize
        :return: Document with Spans created by the Tokenizer.
        """
        assert sdk_isinstance(document, Document)

        if document.text is None:
            return document

        t0 = time.monotonic()
        bbox_keys = sorted([int(x) for x in list(document.get_bbox().keys())])

        if len(bbox_keys) == 0:
            logger.info(
                f'{document} has no characters bboxes. The verifications of the bboxes will not be applied '
                f'for the creation of Spans by the Tokenizer.'
            )

        spans_info = regex_matches(document.text, self.regex)

        # for each Span, create an Annotation
        for span_info in spans_info:

            if len(bbox_keys) > 0 and (
                span_info['start_offset'] not in bbox_keys or span_info['end_offset'] - 1 not in bbox_keys
            ):
                logger.error(
                    f'Regex {span_info["regex_used"]} created span '
                    f'>>{document.text[span_info["start_offset"]:span_info["end_offset"]]}<<'
                    f'with start_offset or end_offset which is not part of the document bbox.'
                )
                continue

            span = Span(
                start_offset=span_info['start_offset'],
                end_offset=span_info['end_offset']
                # , created_by=self.__repr__
            )

            try:
                _ = Annotation(
                    document=document,
                    annotation_set=document.no_label_annotation_set,
                    label=document.project.no_label,
                    label_set=document.project.no_label_set,
                    category=document.category,
                    is_correct=False,
                    revised=False,
                    spans=[span],
                )
            except ValueError as e:
                logger.info(f'Tokenized Annotation for {span}(>>{span.offset_string}<<) not created, because of {e}.')
                continue

        self.processing_steps.append(ProcessingStep(self, document, time.monotonic() - t0))

        return document


class WhitespaceTokenizer(RegexTokenizer):
    """
    Tokenizer based on whitespaces.

    Example:
        "street Name 1-2b," -> "street", "Name", "1-2b,"

    """

    def __init__(self):
        """Initialize the WhitespaceTokenizer."""
        super().__init__(regex=r'[^ \n\t\f]+')


class WhitespaceNoPunctuationTokenizer(RegexTokenizer):
    """
    Tokenizer based on whitespaces without punctuation.

    Example:
        "street Name 1-2b," -> "street", "Name", "1-2b"

    """

    def __init__(self):
        """Initialize the WhitespaceNoPunctuationTokenizer."""
        super().__init__(regex=r'[^ \n\t\f\,\.\;]+')


class ConnectedTextTokenizer(RegexTokenizer):
    r"""
    Tokenizer based on text connected by 1 whitespace.

    Example:
        r"This is \na description. Occupies a paragraph." -> "This is", "a description. Occupies a paragraph."

    """

    def __init__(self):
        """Initialize the ConnectedTextTokenizer."""
        super().__init__(regex=r'(?:(?:[^ \t\n]+(?:[ \t][^ \t\n\:\,\.\!\?\-\_]+)*)+)')


class ColonPrecededTokenizer(RegexTokenizer):
    """
    Tokenizer based on text preceded by colon.

    Example:
        "write to: name" -> "name"

    """

    def __init__(self):
        """Initialize the ColonPrecededTokenizer."""
        super().__init__(regex=r':[ \t]((?:[^ \t\n\:\,\!\?\_]+(?:[ \t][^ \t\n\:\!\?\_]+)*)+)')


class CapitalizedTextTokenizer(RegexTokenizer):
    """
    Tokenizer based on capitalized text.

    Example:
        "Company is Company A&B GmbH now" -> "Company A&B GmbH"

    """

    def __init__(self):
        """Initialize the CapitalizedTextTokenizer."""
        super().__init__(regex=r'(?:[A-ZÄÜÖß][a-zA-Z&äöü]+(?=\s[A-ZÄÜÖß])(?:\s[A-Z&ÄÜÖß][a-zA-Z&äöü]+)+)')


class NonTextTokenizer(RegexTokenizer):
    """
    Tokenizer based on non text - numbers and separators.

    Example:
        "date 01. 01. 2022" -> "01. 01. 2022"

    """

    def __init__(self):
        """Initialize the NonTextTokenizer."""
        super().__init__(regex=r'(?:(?:[A-Z\d]+[:\/. -]{0,2}\n?)+)')


class NumbersTokenizer(RegexTokenizer):
    """
    Tokenizer based on numbers.

    Example:
        "N. 1242022 123 " -> "1242022 123"

    """

    def __init__(self):
        """Initialize the NumbersTokenizer."""
        super().__init__(regex=r'\s((?:[\d+][ ]?)+)\s')


class LineUntilCommaTokenizer(RegexTokenizer):
    r"""
    Tokenizer based on text preceded by colon.

    Example:
        "\n     Company und A&B GmbH,\n" -> "Company und A&B GmbH"

    """

    def __init__(self):
        """Within a line match everything until ','."""
        super().__init__(regex=r'\n\s*([^.]*),\n')


class RegexMatcherTokenizer(ListTokenizer):
    """Applies a list of Tokenizers and then uses a generic regex approach to match the remaining unmatched Spans."""

    def fit(self, category: Category):
        """Call fit on all tokenizers."""
        assert sdk_isinstance(category, Category)

        if not category.documents:
            raise ValueError(f"Category {category.__repr__()} has no training documents.")

        for tokenizer in self.tokenizers:
            tokenizer.fit(category)

        documents = category.documents()

        eval_list = []
        for document in documents:
            # Load Annotations before doing tokenization.
            document.annotations()

            virtual_doc = Document(
                text=document.text,
                bbox=document.get_bbox(),
                project=document.project,
                category=document.category,
                pages=document.pages,
            )
            self.tokenize(virtual_doc)
            df = compare(document, virtual_doc)
            eval_list.append(df)

        if not eval_list:
            raise ValueError('There are no evaluations of the fitted Tokenizer.')

        # Get unmatched Spans.
        df = pd.concat(eval_list)
        spans_not_found_by_tokenizer = df[(df['is_correct']) & (df['is_found_by_tokenizer'] == 0)]
        annotations_not_found_by_tokenizer = [
            x
            for document in documents
            for x in document.annotations()
            if x.id_ in list(spans_not_found_by_tokenizer['id_'].astype(int))
        ]

        # Add Regex tokenizers.
        new_tokenizers = []
        for label in set(x.label for x in annotations_not_found_by_tokenizer):
            label_annotations = [x for x in annotations_not_found_by_tokenizer if x.label == label]
            if label_annotations:
                regexes = label.find_regex(category=category, annotations=label_annotations)
                for regex in regexes:
                    new_tokenizers.append(RegexTokenizer(regex))

        self.tokenizers += new_tokenizers
        logger.info(f'Added {new_tokenizers} to {self}.')
