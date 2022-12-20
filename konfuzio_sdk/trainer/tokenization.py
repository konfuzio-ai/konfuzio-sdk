"""Tokenizers that use byte pair encoding or spaCy NLP package, and various utility functions for Tokenizers."""
import collections
import logging
from typing import Dict, List, Tuple, Union

import spacy
from spacy.matcher import PhraseMatcher
from spacy.language import Language

from konfuzio_sdk.data import Category
import transformers

logger = logging.getLogger(__name__)


class Tokenizer:
    """Converts a string into a list of tokens."""

    def get_entities(self, text: str):
        """
        Given a string use the tokenizer to tokenize the string.

        Returns the entities and their start and end offsets as a sorted (by start_offset) list of dictionaries.
        """
        raise NotImplementedError

    def get_tokens(self, text: str):
        """Similar to get_entities but only returns the entities and not their start/end offsets."""
        raise NotImplementedError


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

        if isinstance(counter, collections.Counter):
            self._stoi, self._itos = self._create_vocab_from_counter(
                counter, min_freq, max_size, unk_token, pad_token, special_tokens
            )
        else:
            # if creating vocab from dict these cannot be set as they have no effect
            assert min_freq == 1
            assert max_size is None
            self._stoi, self._itos = self._create_vocab_from_dict(counter, unk_token, pad_token, special_tokens)

        if unk_token is not None:
            self.unk_idx = self.stoi(unk_token)
        if pad_token is not None:
            self.pad_idx = self.stoi(pad_token)

    def __len__(self):
        """Allow us to do len(Vocab) to get the length of the vocabulary."""
        return len(self._itos)

    def _create_vocab_from_counter(
        self, counter, min_freq, max_size, unk_token, pad_token, special_tokens
    ) -> Tuple[Dict[str, int], List[str]]:
        """
        Handle the actual vocabulary creation.

        Tokens that appear less than min_freq times are ignored
        Once the vocabulary reaches max size, no more tokens are added
        `unk_token` is the token used to replace tokens not in the vocabulary
        `pad_token` is used to pad sequences
        `special_tokens` are other tokens we want appended to the start of our vocabulary, i.e. start of sequence tokens
        """
        stoi = dict()

        if unk_token is not None:
            stoi[unk_token] = len(stoi)
        if pad_token is not None:
            stoi[pad_token] = len(stoi)
        for special_token in special_tokens:
            assert special_token not in stoi
            stoi[special_token] = len(stoi)

        for token, count in counter.most_common(max_size):
            if count >= min_freq:
                if token not in stoi:
                    stoi[token] = len(stoi)
            else:
                break

        itos = list(stoi.keys())

        assert len(stoi) > 0, 'Created vocabulary is empty!'
        assert max_size is None or len(stoi) <= max_size, 'Created vocabulary is larger than max size'
        assert len(stoi) == len(itos), 'Created str -> int vocab len is not the same size as the int -> str vocab len'

        return stoi, itos

    def _create_vocab_from_dict(
        self, counter, unk_token, pad_token, special_tokens
    ) -> Tuple[Dict[str, int], List[str]]:
        """Handle vocabulary creation when we already have a stoi dictionary."""
        if unk_token is not None:
            assert unk_token in counter
        if pad_token is not None:
            assert pad_token in counter
        for special_token in special_tokens:
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


def build_text_vocab(
    categories: List[Category], tokenizer: Tokenizer, min_freq: int = 1, max_size: int = None
) -> Vocab:
    """Build a vocabulary over the document text."""
    logger.info('building text vocab')

    counter = collections.Counter()

    # loop over projects and documents updating counter using the tokens in each document
    for category in categories:
        for document in category.documents():
            tokens = tokenizer.get_tokens(document.text)
            counter.update(tokens)

    assert len(counter) > 0, 'Did not find any tokens when building the text vocab!'

    # create the vocab
    text_vocab = Vocab(counter, min_freq, max_size)

    return text_vocab


def build_template_category_vocab(categories: List[Category]) -> Vocab:
    """Build a vocabulary over the categories of each annotation."""
    logger.info('building category vocab')

    counter = collections.Counter(NO_CATEGORY=0)

    counter.update([str(category.id_) for category in categories])

    template_vocab = Vocab(
        counter, min_freq=1, max_size=None, unk_token=None, pad_token=None, special_tokens=['NO_CATEGORY']
    )

    assert len(template_vocab) > 0, 'Did not find any categories when building the category vocab!'

    # NO_CATEGORY should be label zero so we can avoid calculating accuracy over it later
    assert template_vocab.stoi('NO_CATEGORY') == 0

    return template_vocab


class BPETokenizer(Tokenizer):
    """Tokenizes text using byte-pair encoding models from the hugginface/transformers library."""

    def __init__(self, tokenizer_name: str = 'bert-base-german-cased'):
        """Get the pre-trained BPE tokenizer."""
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

    def get_entities(self, text: str) -> List[Dict[str, Union[str, int]]]:
        """
        Given a string use use the BPE tokenizer to tokenize the string.

        Returns the entities and their start and end offsets as a sorted (by start_offset) list of dictionaries.
        """
        encoded_text = self.tokenizer.encode_plus(
            text, return_offsets_mapping=True, return_token_type_ids=False, return_attention_mask=False, truncation=True
        )
        tokens = self.tokenizer.convert_ids_to_tokens(encoded_text['input_ids'])
        offsets = encoded_text['offset_mapping']

        # convert to list of dictionaries as more commonly used in the training package
        entities = [
            {'offset_string': text, 'start_offset': start, 'end_offset': end}
            for text, (start, end) in zip(tokens, offsets)
        ]

        return entities

    def get_tokens(self, text: str) -> List[str]:
        """Similar to get_entities but only returns the entities and not their start/end offsets."""
        ids = self.tokenizer.encode(text, truncation=True)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens


class SpacyTokenizer(Tokenizer):
    """Tokenizes text using a spaCy model."""

    def __init__(self, spacy_model_name: str = 'de_core_news_sm'):
        """Load a spacy model."""
        self.spacy_model = self._get_spacy_model(spacy_model_name)

    @staticmethod
    def _get_spacy_model(spacy_model_name: str) -> Language:
        """
        Load a spacy model given a string, throw an IO error if not installed/does not exist.

        Returns a spacy language model.
        """
        try:
            spacy_model = spacy.load(spacy_model_name)
        except IOError:
            raise IOError(f'Model not found, please install it with `python -m spacy download {spacy_model_name}`')
        return spacy_model

    def get_entities(self, text: str) -> List[Dict[str, Union[str, int]]]:
        """
        Given a string use the spacy model to tokenize the string.

        Returns the entities and their start and end offsets as a sorted (by start_offset) list of dictionaries.
        """
        entities = set()
        doc = self.spacy_model(text)

        for token in doc:
            token_txt = token.text
            if token_txt.strip() == '':  # skip whitespace
                continue
            start_char = token.idx
            end_char = start_char + len(token_txt)
            entities.add((token_txt, start_char, end_char))

        entities = sorted(entities, key=lambda x: x[1])  # sort by their start offsets

        # convert to list of dictionaries as more commonly used in the training package
        entities = [{'offset_string': text, 'start_offset': start, 'end_offset': end} for text, start, end in entities]

        return entities

    def get_tokens(self, text: str) -> List[str]:
        """Similar to get_entities but only returns the entities and not their start/end offsets."""
        doc = self.spacy_model(text)
        tokens = [token.text for token in doc if token.text.strip() != '']
        return tokens


class PhraseMatcherTokenizer(SpacyTokenizer):
    """Tokenizes text using a spaCy phrase matcher."""

    def __init__(self, categories: List[Category], spacy_model_name: str = 'de_core_news_sm'):
        """Get the spacy model and trains the phrase matcher."""
        self.spacy_model = self._get_spacy_model(spacy_model_name)
        self.phrase_matcher = self._train_phrase_matcher(categories)

    def _train_phrase_matcher(self, categories: List[Category]) -> PhraseMatcher:
        """
        Train a spaCy phrase matcher.

        Given an iterable of documents train a spaCy PhraseMatcher on each of the labels within the annotations
        belonging to the documents.

        Returns the phrase matcher.
        """
        logger.info('Getting phrase matcher training data')

        # collect all examples of each label within the training set - used to train the phrase matcher
        train_dataset = collections.defaultdict(set)

        for category in categories:
            for document in category.documents():
                for span in document.spans():
                    train_dataset[span.annotation.label.name].add(span.offset_string)

        logger.info('Creating phrase matcher')
        # create instance of a phrase matcher
        phrase_matcher = PhraseMatcher(self.spacy_model.vocab, attr='shape')

        logger.info('Training phrase matcher')
        for label in train_dataset.keys():
            # get examples to train phrase matcher
            examples = train_dataset[label]
            # build label-attr-phrase-matcher
            phrase_matcher.add(label, list(self.spacy_model.tokenizer.pipe(examples)))

        self.phrase_matcher = phrase_matcher
        return phrase_matcher

    def get_entities(self, text: str) -> List[Dict[str, Union[str, int]]]:
        """
        Given a string use the phrase matcher and spacy model to tokenize the string.

        Returns the entities and their start and end offsets as a sorted (by start_offset) list of dictionaries.
        """
        entities = set()
        doc = self.spacy_model(text)

        # by default the phrase matcher's start and end are the token indices
        # we convert these into character offsets from the beginning of the string
        for match_id, start_tok, end_tok in self.phrase_matcher(doc):
            span = [token.idx for token in doc[start_tok:end_tok]]  # .idx gets the character offset to start of token
            start_char = span[0]
            end_char = span[-1] + len(doc[end_tok - 1])  # end tok is one after the actual last token
            matched_text = doc[start_tok:end_tok].text  # get actual text (str) between start and end
            if matched_text.strip() == '':  # skip whitespace
                continue
            assert matched_text == text[start_char:end_char]  # ensure it matches the given text str
            entities.add((matched_text, start_char, end_char))

        for token in doc:
            token_txt = token.text
            if token_txt.strip() == '':  # skip whitespace
                continue
            start_char = token.idx
            end_char = start_char + len(token_txt)
            entities.add((token_txt, start_char, end_char))

        entities = sorted(entities, key=lambda x: x[1])  # sort by their start offsets

        # convert to list of dictionaries as more commonly used in the training package
        entities = [{'offset_string': text, 'start_offset': start, 'end_offset': end} for text, start, end in entities]

        return entities

    def get_tokens(self, text: str) -> List[str]:
        """Similar to get_entities but only returns the entities and not their start/end offsets."""
        entities = self.get_entities(text)
        tokens = [e['offset_string'] for e in entities]
        return tokens


# def get_tokenizer(tokenizer_name: str, categories: List[Category], *args, **kwargs) -> Tokenizer:
#     """Get a Tokenizer based on a string. Some Tokenizers need a list of projects to build themselves."""
#     if tokenizer_name == 'phrasematcher':
#         tokenizer = PhraseMatcherTokenizer(categories)
#     else:
#         try:  # try and get a bpe tokenizer from huggingface
#             tokenizer = BPETokenizer(tokenizer_name)
#         except OSError:
#             raise ValueError(f'{tokenizer_name} is not a valid BPE tokenizer!')
#
#     return tokenizer
