"""Generic tokenizer."""

import abc
import collections
import logging
from typing import List, Union, Tuple, Dict, Optional
from copy import deepcopy

import pandas as pd

from konfuzio_sdk.data import Document, Span, Page
from konfuzio_sdk.evaluate import compare, ExtractionEvaluation
from konfuzio_sdk.utils import sdk_isinstance

logger = logging.getLogger(__name__)


class Vocab:
    """
    Class to handle a vocabulary, a mapping between strings and their corresponding integer values.

    Vocabulary must be created with a counter where each key is a token and each value is the number
    of times that tokens appears in the training dataset.
    """

    def __init__(
        self,
        counter: Union[collections.Counter, dict],
        min_freq: int = 1,
        max_size: int = None,
        unk_token: str = '<unk>',
        pad_token: str = '<pad>',
        special_tokens=None,
    ):
        """Initialize the Vocab object and builds the vocabulary mappings."""
        if special_tokens is None:
            special_tokens = []
        assert min_freq >= 1

        self.counter = counter
        self.min_freq = min_freq
        self.max_size = max_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.special_tokens = special_tokens

        if unk_token is not None:
            assert unk_token not in special_tokens
        if pad_token is not None:
            assert pad_token not in special_tokens

        if isinstance(self.counter, collections.Counter):
            self._stoi, self._itos = self._create_vocab_from_counter()
        else:
            # if creating vocab from dict these cannot be set as they have no effect
            assert min_freq == 1
            assert max_size is None
            self._stoi, self._itos = self._create_vocab_from_dict(self.counter)

        if unk_token is not None:
            self.unk_idx = self.stoi(unk_token)
        if pad_token is not None:
            self.pad_idx = self.stoi(pad_token)

        assert len(self) > 0, 'Did not find any categories when building the category vocab!'
        # NO_CATEGORY ('0') should be label zero so we can avoid calculating accuracy over it later

    def __len__(self):
        """Allow us to do len(Vocab) to get the length of the vocabulary."""
        return len(self._itos)

    def _create_vocab_from_counter(self) -> Tuple[Dict[str, int], List[str]]:
        """
        Handle the actual vocabulary creation.

        Tokens that appear less than min_freq times are ignored
        Once the vocabulary reaches max size, no more tokens are added
        `unk_token` is the token used to replace tokens not in the vocabulary
        `pad_token` is used to pad sequences
        `special_tokens` are other tokens we want appended to the start of our vocabulary, i.e. start of sequence tokens
        """
        stoi = dict()

        if self.unk_token is not None:
            stoi[self.unk_token] = len(stoi)
        if self.pad_token is not None:
            stoi[self.pad_token] = len(stoi)
        for special_token in self.special_tokens:
            assert special_token not in stoi
            stoi[special_token] = len(stoi)

        for token, count in self.counter.most_common(self.max_size):
            if count >= self.min_freq:
                if token not in stoi:
                    stoi[token] = len(stoi)
            else:
                break

        itos = list(stoi.keys())

        assert len(stoi) > 0, 'Created vocabulary is empty!'
        assert self.max_size is None or len(stoi) <= self.max_size, 'Created vocabulary is larger than max size'
        assert len(stoi) == len(itos), 'Created str -> int vocab len is not the same size as the int -> str vocab len'

        return stoi, itos

    def _create_vocab_from_dict(self, counter) -> Tuple[Dict[str, int], List[str]]:
        """Handle vocabulary creation when we already have a stoi dictionary."""
        if self.unk_token is not None:
            assert self.unk_token in counter
        if self.pad_token is not None:
            assert self.pad_token in counter
        for special_token in self.special_tokens:
            assert special_token in counter

        stoi = counter

        itos = list(stoi.keys())

        return stoi, itos

    def stoi(self, token: str) -> int:
        """
        Convert a token (str) into its corresponding integer value from the vocabulary.

        If the token is not in the vocabulary, returns the integer value of the unk_token
        If unk_token is set to None, throws an error
        """
        assert isinstance(token, str), f'Input to vocab.stoi should be str, got {type(token)}'

        if token in self._stoi:
            return self._stoi[token]
        else:
            assert self.unk_token is not None, f'token {token} is not in the vocab and unk_token = None!'
            return self._stoi[self.unk_token]

    def itos(self, index: int) -> str:
        """
        Convert an integer into its corresponding token (str) from the vocabulary.

        If the integer value is outside of the vocabulary range, throws an error.
        """
        assert isinstance(index, int), f'Input to vocab.itos should be an integer, got {type(index)}'
        assert index >= 0, f'Input to vocab.itos should be a non-negative, got {index}'
        assert index < len(self._itos), f'Input index out of range, should be <{len(self._itos)}, got {index}'

        return self._itos[index]

    def get_tokens(self) -> List[str]:
        """Return the list of tokens (str) in the vocab."""
        return self._itos[:]

    def get_indexes(self) -> List[int]:
        """Return the list of indexes (int) in the vocab."""
        return list(self._stoi.values())

    def numericalize(
        self, page_or_document: Union[Page, Document], max_length: Optional[int] = None, pad: bool = True
    ) -> Union[Page, Document]:
        """Convert each Span of the Document or Page into its corresponding integer value from the vocabulary."""
        # todo assert we have a valid token (e.g '\n\n\n' results in tok = [])
        # if not tok:
        #    logger.info(f'[WARNING] The token resultant from page {i} is empty. Page text: {txt}.')
        text_encoded: List[int]
        if isinstance(page_or_document, Document):
            document = page_or_document
            document.text_encoded = [self.stoi(span.offset_string) for span in document.spans()][:max_length]
        elif isinstance(page_or_document, Page):
            page = page_or_document
            page.text_encoded = [self.stoi(span.offset_string) for span in page.spans()][:max_length]
        else:
            raise NotImplementedError
        if pad and not page_or_document.text_encoded:
            page_or_document.text_encoded = [self.pad_idx]
        return page_or_document


class ProcessingStep:
    """Track runtime of Tokenizer functions."""

    def __init__(self, tokenizer: 'AbstractTokenizer', document: Document, runtime: float):
        """Initialize the processing step."""
        self.tokenizer = tokenizer
        self.document = document
        self.runtime = runtime

    def eval_dict(self):
        """Return any information needed to evaluate the ProcessingStep."""
        step_eval = {
            'tokenizer_name': str(self.tokenizer),
            'document_id': self.document.id_ or self.document.copy_of_id,
            'number_of_pages': self.document.number_of_pages,
            'runtime': self.runtime,
        }
        return step_eval


class AbstractTokenizer(metaclass=abc.ABCMeta):
    """Abstract definition of a Tokenizer."""

    processing_steps = []

    def __repr__(self):
        """Return string representation of the class."""
        return f"{self.__class__.__name__}"

    def __eq__(self, other) -> bool:
        """Check if two Tokenizers are the same."""
        return hash(self) == hash(other)

    def __call__(self, document: Document) -> Document:
        """Tokenize the Document with this Tokenizer."""
        return self.tokenize(document=document)

    @abc.abstractmethod
    def __hash__(self):
        """Get unique hash for Tokenizer."""

    @abc.abstractmethod
    def found_spans(self, document: Document) -> List[Span]:
        """Find all Spans in a Document that can be found by a Tokenizer."""

    @abc.abstractmethod
    def tokenize(self, document: Document) -> Document:
        """
        Create Annotations with 1 Span based on the result of the Tokenizer.

        :param document: Document to tokenize, can have been tokenized before
        :return: Document with Spans created by the Tokenizer.
        """

    def evaluate(self, document: Document) -> pd.DataFrame:
        """
        Compare a Document with its tokenized version.

        :param document: Document to evaluate
        :return: Evaluation DataFrame
        """
        assert sdk_isinstance(document, Document)
        document.annotations()  # Load Annotations before doing tokenization

        virtual_doc = deepcopy(document)
        self.tokenize(virtual_doc)
        evaluation = compare(document, virtual_doc, use_view_annotations=False, ignore_below_threshold=False)
        logger.warning(
            f'{evaluation["tokenizer_true_positive"].sum()} of {evaluation["is_correct"].sum()} corrects'
            f' Spans are found by Tokenizer'
        )
        return evaluation

    def evaluate_dataset(self, dataset_documents: List[Document]) -> ExtractionEvaluation:
        """
        Evaluate the tokenizer on a dataset of documents.

        :param dataset_documents: Documents to evaluate
        :return: ExtractionEvaluation instance
        """
        eval_list = []
        for document in dataset_documents:
            assert sdk_isinstance(document, Document), f"Invalid document type: {type(document)}. Should be Document."
            document.annotations()  # Load Annotations before doing tokenization
            virtual_doc = deepcopy(document)
            self.tokenize(virtual_doc)
            eval_list.append((document, virtual_doc))
        return ExtractionEvaluation(eval_list, use_view_annotations=False, ignore_below_threshold=False)

    def missing_spans(self, document: Document) -> List[Span]:
        """
        Apply a Tokenizer on a Document and find all Spans that cannot be found.

        Use this approach to sequentially work on remaining Spans after a Tokenizer ran on a List of Documents.

        :param document: A Document

        :return: A list containing all missing Spans.

        """
        self.found_spans(document)
        missing_spans_list = [span for span in document.spans(use_correct=True) if span.regex_matching == []]

        return missing_spans_list

    def get_runtime_info(self) -> pd.DataFrame:
        """
        Get the processing runtime information as DataFrame.

        :return: processing time Dataframe containing the processing duration of all steps of the tokenization.
        """
        data = [x.eval_dict() for x in self.processing_steps]
        return pd.DataFrame(data)

    def lose_weight(self):
        """Delete processing steps."""
        self.processing_steps = []


class ListTokenizer(AbstractTokenizer):
    """Use multiple tokenizers."""

    def __init__(self, tokenizers: List['AbstractTokenizer']):
        """Initialize the list of tokenizers."""
        self.tokenizers = list(dict.fromkeys(tokenizers))
        self.processing_steps = []

    def __eq__(self, other) -> bool:
        """Compare ListTokenizer with another Tokenizer."""
        if type(other) is ListTokenizer:
            return self.tokenizers == other.tokenizers
        else:
            return False

    def __hash__(self):
        """Get unique hash for ListTokenizer."""
        return hash(tuple(self.tokenizers))

    def _tokenize(self, document: Document) -> List[Span]:
        raise NotImplementedError

    def tokenize(self, document: Document) -> Document:
        """Run tokenize in the given order on a Document."""
        assert sdk_isinstance(document, Document)

        for tokenizer in self.tokenizers:
            # todo: running multiple tokenizers on one document
            #  should support that multiple Tokenizers can create identical Spans
            tokenizer.tokenize(document)
            if tokenizer.processing_steps:
                self.processing_steps.append(tokenizer.processing_steps[-1])

        return document

    def found_spans(self, document: Document) -> List[Span]:
        """Run found_spans in the given order on a Document."""
        found_spans_list = []
        for tokenizer in self.tokenizers:
            found_spans_list += tokenizer.found_spans(document)
        return found_spans_list

    def span_match(self, span: 'Span') -> bool:
        """Run span_match in the given order."""
        for tokenizer in self.tokenizers:
            if tokenizer.span_match(span):
                return True
        return False

    def lose_weight(self):
        """Delete processing steps."""
        self.processing_steps = []
        for tokenizer in self.tokenizers:
            tokenizer.lose_weight()
