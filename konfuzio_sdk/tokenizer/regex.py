"""Regex tokenizers."""
import logging
from typing import List


from konfuzio_sdk.data import Document, Category, Span
from konfuzio_sdk.regex import regex_matches
from konfuzio_sdk.tokenizer.base import AbstractTokenizer
from konfuzio_sdk.utils import sdk_isinstance

logger = logging.getLogger(__name__)


class RegexTokenizer(AbstractTokenizer):
    """Tokenizer based on a single regex."""

    def __init__(self, regex: str):
        """Initialize the RegexTokenizer."""
        self.regex = regex
        self.processing_steps = []

    def __repr__(self):
        """Return string representation of the class."""
        return f"{self.__class__.__name__}: {repr(self.regex)}"

    def __hash__(self):
        """Get unique hash for RegexTokenizer."""
        return hash(repr(self.regex))

    def __eq__(self, other) -> bool:
        """Compare RegexTokenizer with another Tokenizer."""
        return hash(self) == hash(other)

    def fit(self, category: Category):
        """Fit the tokenizer accordingly with the Documents of the Category."""
        assert sdk_isinstance(category, Category)
        return self

    def _tokenize(self, document: Document) -> List[Span]:
        """
        Given a Document use the regex to tokenize the text of the Document.

        Returns the Spans as a sorted (by start_offset) list.
        """
        spans = []
        # do not keep the full regex match as we will see many matches whitespaces as pre or suffix
        for span_info in regex_matches(document.text, self.regex, keep_full_match=False):
            span = Span(start_offset=span_info['start_offset'], end_offset=span_info['end_offset'])
            span.regex_matching.append(self)
            if span not in spans:  # do not use duplicated spans  # todo add test
                spans.append(span)
        return spans

    def span_match(self, span: 'Span') -> bool:
        """Check if Span is detected by Tokenizer."""
        if self in span.regex_matching:
            return True
        else:
            relevant_text_slice = span.annotation.document.text
            for span_info in regex_matches(relevant_text_slice, self.regex, keep_full_match=False):
                span_info_offsets = (span_info['start_offset'], span_info['end_offset'])
                if span_info_offsets == (span.start_offset, span.end_offset):
                    span.regex_matching.append(self)
                    return True
        return False

    def found_spans(self, document: Document) -> List[Span]:
        """
        Find Spans found by the Tokenizer and add Tokenizer info to Span.

        :param document: Document with Annotation to find.
        :return: List of Spans found by the Tokenizer.
        """
        assert sdk_isinstance(document, Document)
        if document.text is None:
            raise NotImplementedError(f'{document} text is None.')

        document_spans = {(span.start_offset, span.end_offset): span for span in document.spans()}
        found_spans_list = []
        for span_info in regex_matches(document.text, self.regex, keep_full_match=False):
            span_offsets = (span_info['start_offset'], span_info['end_offset'])
            if span_offsets in document_spans:
                found_spans_list.append(document_spans[span_offsets])
                if self not in document_spans[span_offsets].regex_matching:
                    document_spans[span_offsets].regex_matching.append(self)

        return found_spans_list


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


class ColonOrWhitespacePrecededTokenizer(RegexTokenizer):
    """
    Tokenizer based on text preceded by colon.

    Example:
        "write to: name" -> "name"

    """

    def __init__(self):
        """Initialize the ColonPrecededTokenizer."""
        super().__init__(
            regex=r'[ :][ \t](?P<ColonOrWhitespacePreceded>(?:[^ \t\n\:\,\!\?\_]+(?:[ \t][^ \t\n\:\!\?\_]+)*)+)'
        )


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
