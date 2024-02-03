"""Tokenizers that use byte pair encoding or spaCy NLP package, and various utility functions for Tokenizers."""
import collections
import logging
from typing import Any, List

from konfuzio_sdk.data import Category, Document, Span
from konfuzio_sdk.extras import SpacyLanguage, SpacyPhraseMatcher, spacy, transformers
from konfuzio_sdk.tokenizer.base import AbstractTokenizer, Vocab
from konfuzio_sdk.utils import sdk_isinstance

logger = logging.getLogger(__name__)

# these modules are WIP and do not have testing yet.

HF_ALLOWED_TOKENIZERS = [
    transformers.BertTokenizerFast,  # tokenizer for BERT
    transformers.DistilBertTokenizerFast,  # tokenizer for DistilBERT
    transformers.AlbertTokenizerFast,  # tokenizer for ALBERT
    transformers.MobileBertTokenizerFast,  # tokenizer for MobileBERT
    transformers.ElectraTokenizerFast,  # tokenizer for ELECTRA
]


class Tokenizer(AbstractTokenizer):
    """Base Tokenizer."""

    def __init__(self, tokenizer_name: str):
        """Initialize the RegexTokenizer."""
        self.tokenizer_name = tokenizer_name
        self.processing_steps = []

    def __repr__(self):
        """Return string representation of the class."""
        return f'{self.__class__.__name__}: {repr(self.tokenizer_name)}'

    def __hash__(self):
        """Get unique hash for RegexTokenizer."""
        return hash(repr(self.tokenizer_name))

    def __eq__(self, other) -> bool:
        """Compare RegexTokenizer with another Tokenizer."""
        return hash(self) == hash(other)

    def fit(self, category: Category):
        """Fit the tokenizer accordingly with the Documents of the Category."""
        assert sdk_isinstance(category, Category)
        return self

    def _tokenize(self, document: Document) -> List[Span]:
        """
        Given a Document use the tokenizer to tokenize the text of the Document.

        Returns the Spans as a list.
        """
        return []

    def found_spans(self, document: Document) -> List[Span]:
        """
        Find Spans found by the Tokenizer and add Tokenizer info to Span.

        :param document: Document with Annotation to find.
        :return: List of Spans found by the Tokenizer.
        """
        raise NotImplementedError


class BPETokenizer(Tokenizer):
    """Tokenizes text using byte-pair encoding models from the hugginface/transformers library."""

    def __init__(self, tokenizer_name: str = 'bert-base-german-cased'):
        """Get the pre-trained BPE tokenizer."""
        super().__init__(tokenizer_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.vocab = self._create_vocab()

    def _create_vocab(self) -> Vocab:
        """Get vocabulary for the pre-trained BPE tokenizer."""
        stoi = self.tokenizer.get_vocab()
        unk_token = self.tokenizer.unk_token
        pad_token = self.tokenizer.pad_token
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        mask_token = self.tokenizer.mask_token
        additional_tokens = self.tokenizer.additional_special_tokens
        special_tokens = [bos_token, eos_token, cls_token, sep_token, mask_token]
        special_tokens.extend(additional_tokens)
        special_tokens = [t for t in special_tokens if t is not None]
        vocab = Vocab(stoi, unk_token=unk_token, pad_token=pad_token, special_tokens=special_tokens)
        return vocab

    def _tokenize(self, document: Document) -> List[Span]:
        """
        Given a Document use the BPE tokenizer to tokenize the text of the Document.

        Returns the Spans as a list.
        """
        spans = []
        text = document.text

        encoded_text = self.tokenizer.encode_plus(
            text, return_offsets_mapping=True, return_token_type_ids=False, return_attention_mask=False, truncation=True
        )
        # tokens = self.tokenizer.convert_ids_to_tokens(encoded_text['input_ids'])
        # convert to list of dictionaries as more commonly used in the training package
        for start, end in encoded_text['offset_mapping']:
            span = Span(start_offset=start, end_offset=end)
            if span not in spans:  # do not use duplicated spans  # todo add test
                spans.append(span)

        return spans


class SpacyTokenizer(Tokenizer):
    """Tokenizes text using a spaCy model."""

    def __init__(self, spacy_model_name: str = 'de_core_news_sm'):
        """Load a spacy model."""
        super().__init__(spacy_model_name)
        self.spacy_model = self._get_spacy_model(spacy_model_name)

    @staticmethod
    def _get_spacy_model(spacy_model_name: str) -> SpacyLanguage:
        """
        Load a spacy model given a string, throw an IO error if not installed/does not exist.

        Returns a spacy language model.
        """
        try:
            spacy_model = spacy.load(spacy_model_name)
        except IOError:
            raise IOError(f'Model not found, please install it with `python -m spacy download {spacy_model_name}`')
        return spacy_model

    def _tokenize(self, document: Document) -> List[Span]:
        """
        Given a Document use the spacy model to tokenize the text of the Document.

        Returns the Spans as a list.
        """
        spans = []
        text = document.text
        spacy_doc = self.spacy_model(text)

        for token in spacy_doc:
            if token.text.strip() == '':  # skip whitespace
                continue
            start_char = token.idx
            end_char = start_char + len(token.text)
            span = Span(start_offset=start_char, end_offset=end_char)
            # span.regex_matching.append(self)
            if span not in spans:  # do not use duplicated spans  # todo add test
                spans.append(span)

        return spans


class PhraseMatcherTokenizer(SpacyTokenizer):
    """Tokenizes text using a spaCy phrase matcher."""

    def __init__(self, documents: List[Document], spacy_model_name: str = 'de_core_news_sm'):
        """Get the spacy model and trains the phrase matcher."""
        super().__init__(spacy_model_name)
        self.phrase_matcher = self._train_phrase_matcher(documents)

    def _train_phrase_matcher(self, documents: List[Document]) -> SpacyPhraseMatcher:
        """
        Train a spaCy phrase matcher.

        Given an iterable of documents train a spaCy PhraseMatcher on each of the labels within the annotations
        belonging to the documents.

        Returns the phrase matcher.
        """
        logger.info('Getting phrase matcher training data')

        # collect all examples of each label within the training set - used to train the phrase matcher
        train_dataset = collections.defaultdict(set)

        for document in documents:
            for span in document.spans():
                train_dataset[span.annotation.label.name].add(span.offset_string)

        logger.info('Creating phrase matcher')
        # create instance of a phrase matcher
        phrase_matcher = SpacyPhraseMatcher(self.spacy_model.vocab, attr='shape')

        logger.info('Training phrase matcher')
        for label in train_dataset.keys():
            # get examples to train phrase matcher
            examples = train_dataset[label]
            # build label-attr-phrase-matcher
            phrase_matcher.add(label, list(self.spacy_model.tokenizer.pipe(examples)))

        self.phrase_matcher = phrase_matcher
        return phrase_matcher

    def _tokenize(self, document: Document) -> List[Span]:
        """
        Given a Document use the phrase matcher and spacy model to tokenize the text of the Document.

        Returns the Spans as a list.
        """
        spans = []
        text = document.text
        spacy_doc = self.spacy_model(text)

        # by default the phrase matcher's start and end are the token indices
        # we convert these into character offsets from the beginning of the string
        for match_id, start_tok, end_tok in self.phrase_matcher(spacy_doc):
            # .idx gets the character offset to start of token
            spacy_span = [token.idx for token in spacy_doc[start_tok:end_tok]]
            start_char = spacy_span[0]
            end_char = spacy_span[-1] + len(spacy_doc[end_tok - 1])  # end tok is one after the actual last token
            matched_text = spacy_doc[start_tok:end_tok].text  # get actual text (str) between start and end
            if matched_text.strip() == '':  # skip whitespace
                continue
            assert matched_text == text[start_char:end_char]  # ensure it matches the given text str
            span = Span(start_offset=start_char, end_offset=end_char)
            if span not in spans:  # do not use duplicated spans  # todo add test
                spans.append(span)

        for token in spacy_doc:
            if token.text.strip() == '':  # skip whitespace
                continue
            start_char = token.idx
            end_char = start_char + len(token.text)
            span = Span(start_offset=start_char, end_offset=end_char)
            # span.regex_matching.append(self)
            if span not in spans:  # do not use duplicated spans  # todo add test
                spans.append(span)

        return spans

    def tokenize(self, document: Document) -> Document:
        """Tokenize the Document."""
        spans = self._tokenize(document)

        for span in spans:
            _ = Span(start_offset=span.start_offset, end_offset=span.end_offset, document=document)

        return document


class TransformersTokenizer:
    """
    SDK Wrapper for transformers.AutoTokenizer.

    Tokenizes text using the tokenizer returned
    from transformers.AutoTokenizer (must be in the allowed_tokenizers list).

    :param tokenizer_name: The name or path of the pre-trained tokenizer. Default is 'bert-base-german-cased'.

    :return: None

    :raises ValueError: If the provided tokenizer is not in the allowed_tokenizers list.
    """

    def __init__(
        self,
        tokenizer_name: str = 'bert-base-german-cased',
    ) -> None:
        """Initialize the TransformersTokenizer."""
        # Initialize the tokenizer using transformers.AutoTokenizer.from_pretrained()
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=tokenizer_name,
            )
        except Exception:
            raise ValueError(f'Could not load Transformers Tokenizer {self.name}.')

        # Check if the provided tokenizer is in the allowed_tokenizers list
        if not isinstance(self.tokenizer, tuple(HF_ALLOWED_TOKENIZERS)):
            raise ValueError(
                f'The tokenizer {tokenizer_name} is not supported.  \
                Please use a tokenizer that belongs to one of the following classes: {HF_ALLOWED_TOKENIZERS}'
            )

    def __call__(
        self,
        *args,
        max_length: int = 512,
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: str = None,
        **kwargs: Any,
    ) -> Any:
        """
        Call method to invoke the tokenizer.

        :param max_length: Maximum sequence length after tokenization. Default is 512.

        :param padding: 'max_length' pads to the maximum sequence length;
        'longest' pads to the length of the longest sequence. Default is 'max_length'.

        :param truncation: If True, truncate sequences longer than max_length. Default is True.

        :param return_tensors: The type of PyTorch tensors to be returned.
        Default is None so it returns a python list (since we already transform to tensor).

        :param kwargs: Additional keyword arguments to be passed to the tokenizer.

        :return: The result of calling the underlying tokenizer with the provided arguments.
        """
        return self.tokenizer(
            *args,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            **kwargs,
        )
