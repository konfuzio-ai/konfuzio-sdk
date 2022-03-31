"""Regex tokenizers."""
from konfuzio_sdk.data import Annotation, Document, Category, Span
from konfuzio_sdk.regex import regex_matches
from konfuzio_sdk.tokenizer.base import AbstractTokenizer


class RegexTokenizer(AbstractTokenizer):
    """Tokenizer based on a single regex."""

    def __init__(self, regex: str):
        """Initialize the SingleRegexTokenizer."""
        self.regex = regex

    def fit(self, category: Category):
        """Fit the tokenizer accordingly with the Documents of the Category."""
        assert isinstance(category, Category)
        return self

    def tokenize(self, document: Document) -> Document:
        """
        Create Annotations with 1 Span based on the result of the Tokenizer.

        :param document: Document to tokenize
        :return: Document with Spans created by the Tokenizer.
        """
        assert isinstance(document, Document)

        if not document.text:
            return document

        # t0 = time.monotonic()
        spans_info = regex_matches(document.text, self.regex)

        # for each Span, create an Annotation
        for span_info in spans_info:
            span = Span(
                start_offset=span_info['start_offset'],
                end_offset=span_info['end_offset']
                # , created_by=self.__repr__
            )

            # TODO: check that tokenizer does ot create empty spans
            # TODO: check if bboxes of characters are available
            # if document.text[span.start_offset: span.end_offset] == '':  # skip whitespace
            #     logger.error(f'Whitespace entity entered training process.')
            #     continue

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

        # TODO: add processing time to Document
        # document.add_process_step(self.__repr__, time.monotonic() - t0)

        return document


class WhitespaceTokenizer(RegexTokenizer):
    """Tokenizer based on whitespaces."""

    def __init__(self):
        """Initialize the WhitespaceTokenizer."""
        super().__init__(regex=r'[^ \n\t\f]+')


class ConnectedTextTokenizer(RegexTokenizer):
    """Tokenizer based on text connected by 1 whitespace."""

    def __init__(self):
        """Initialize the ConnectedTextTokenizer."""
        super().__init__(regex=r'(?:(?:[^ \t\n]+(?:[ \t][^ \t\n\:\,\.\!\?\-\_]+)*)+)')


class ColonPrecededTokenizer(RegexTokenizer):
    """Tokenizer based on text preceded by colon."""

    def __init__(self):
        """Initialize the ColonPrecededTokenizer."""
        super().__init__(regex=r'(?:(?::[ \t])((?:[^ \t\n\:\,\.\!\?\-\_]+(?:[ \t][^ \t\n\:\,\.\!\?\-\_]+)*)+))')


class CapitalizedTextTokenizer(RegexTokenizer):
    """Tokenizer based on capitalized text."""

    def __init__(self):
        """Initialize the CapitalizedTextTokenizer."""
        super().__init__(regex=r'(?:[A-ZÄÜÖß][a-zA-Z&]+(?=\s[A-ZÄÜÖß])(?:\s[A-Z&ÄÜÖß][a-zA-Z&]+)+)')


class NonTextTokenizer(RegexTokenizer):
    """Tokenizer based on non text - numbers and separators."""

    def __init__(self):
        """Initialize the NonTextTokenizer."""
        super().__init__(regex=r'(?:(?:[A-Z\d]+[:\/. -]*\n?)+)')
