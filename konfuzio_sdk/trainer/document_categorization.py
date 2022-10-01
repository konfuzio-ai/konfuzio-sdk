"""Implements a DocumentModel."""

# import os
import collections
import datetime
import functools
import os
import random
import re
import math

# import sys
import logging
import tempfile
import uuid
from copy import deepcopy
from typing import Union, List, Dict, Tuple
from warnings import warn

# import pathlib

# import cloudpickle
import timm
import torchvision
import PIL.ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from konfuzio_sdk.data import Project, Document, Category
from konfuzio_sdk.evaluate import CategorizationEvaluation
from konfuzio_sdk.trainer.tokenization import Vocab, Tokenizer, BPETokenizer, get_tokenizer

logger = logging.getLogger(__name__)

warn('This module is WIP: https://gitlab.com/konfuzio/objectives/-/issues/9481', FutureWarning, stacklevel=2)

# ensure backwards compatibility of models saved with project name instead of id.
# Can be deleted when old models are adapted or not needed anymore
NAMES_CONVERSION = {
    'Versicherungspolicen Erweitert (Multipolicen, KFZ, Haftpflicht, ..)': 23,
    'Münchener Verein - Demo': 24,
    'Kontoauszug': 34,
    'Rechnung (German)': 39,
    'BU-Versicherung': 43,
    'RV-Verträge - Demo': 44,
    'KONFUZIO_PAYSLIP_TESTS': 46,
    'ZIMDB-ZIAS: Grundbuchauszug': 50,
    'Arztrechnung': 58,
    'Darlehensvertrag - IDS': 60,
    'Syncier - Financial Statement': 61,
    'ID Cards (from dataset) - Demo': 76,
    'Grundbuchauszug': 145,
    'Kaufvertrag': 169,
    'Expose': 170,
    'Flurkarte': 171,
    'Grundriss': 172,
    'Mietvertrag': 173,
    'Teilungserklaerung': 174,
    'WohnNutzflaechBerech': 175,
}


def get_category_name_for_fallback_prediction(category: Union[Category, str]) -> str:
    """Turn a category name to lowercase, remove parentheses along with their contents, and trim spaces."""
    if isinstance(category, Category):
        category_name = category.name.lower()
    elif isinstance(category, str):
        category_name = category.lower()
    else:
        raise NotImplementedError
    parentheses_removed = re.sub(r'\([^)]*\)', '', category_name).strip()
    single_spaces = parentheses_removed.replace("  ", " ")
    return single_spaces


def build_list_of_relevant_categories(training_categories: List[Category]) -> List[List[str]]:
    """Filter for category name variations which correspond to the given categories, starting from a predefined list."""
    relevant_categories = []
    for training_category in training_categories:
        category_name = get_category_name_for_fallback_prediction(training_category)
        relevant_categories.append(category_name)
    return relevant_categories


class FallbackCategorizationModel:
    """A model that predicts a category for a given document."""

    def __init__(self, project: Union[int, Project], *args, **kwargs):
        """Initialize FallbackCategorizationModel."""
        # Go through keyword arguments, and either save their values to our
        # instance, or raise an error.
        if isinstance(project, int):
            self.project = Project(id_=project)
        elif isinstance(project, Project):
            self.project = project
        else:
            raise NotImplementedError

        self.categories = None
        self.name = self.__class__.__name__

        self.evaluation = None

    def fit(self) -> None:
        """Use as placeholder Function."""
        raise NotImplementedError(
            f'{self} uses a fallback logic for categorizing documents, and does not train a classifier.'
        )

    def save(self, output_dir: str, include_konfuzio=True):
        """Use as placeholder Function."""
        raise NotImplementedError(
            f'{self} uses a fallback logic for categorizing documents, this will not save model to disk.'
        )

    def evaluate(self) -> CategorizationEvaluation:
        """Evaluate the full Categorization pipeline on the pipeline's Test Documents."""
        eval_list = []
        for document in self.test_documents:
            predicted_doc = self.categorize(document=document, recategorize=True)
            eval_list.append((document, predicted_doc))

        self.evaluation = CategorizationEvaluation(self.project, eval_list)

        return self.evaluation
        # raise NotImplementedError(
        #     f'{self} uses a fallback logic for categorizing documents, without using Training or Test documents for '
        #     f'evaluation.'
        # )

    def categorize(self, document: Document, recategorize: bool = False, inplace: bool = False) -> Document:
        """Run categorization."""
        if inplace:
            virtual_doc = document
        else:
            virtual_doc = deepcopy(document)
        if (document.category is not None) and (not recategorize):
            logger.info(
                f'In {document}, the category was already specified as {document.category}, so it wasn\'t categorized '
                f'again. Please use recategorize=True to force running the Categorization AI again on this document.'
            )
            return virtual_doc
        elif recategorize:
            virtual_doc.category = None

        relevant_categories = build_list_of_relevant_categories(self.categories)
        found_category_name = None
        doc_text = virtual_doc.text.lower()
        for candidate_category_name in relevant_categories:
            if candidate_category_name in doc_text:
                found_category_name = candidate_category_name
                break

        if found_category_name is None:
            logger.warning(
                f'{self} could not find the category of {document} by using the fallback logic '
                f'with pre-defined common categories.'
            )
            return virtual_doc
        found_category = [
            category
            for category in self.categories
            if get_category_name_for_fallback_prediction(category) in found_category_name
        ][0]
        virtual_doc.category = found_category
        return virtual_doc


class InvertImage:
    """Invert (negate) images."""

    def __call__(self, sample):
        """Apply image invertion."""
        image = PIL.ImageOps.invert(sample)
        return image


class ImagePreProcessing:
    """Define the images pre processing transformations."""

    def __init__(self, transforms: dict = {'target_size': (1000, 1000)}):
        """Collect the transformations to be applied to the images."""
        self.pre_processing_operations = []
        transforms_keys = transforms.keys()

        if 'invert' in transforms_keys and transforms['invert']:
            self.pre_processing_operations.append(InvertImage())

        if 'target_size' in transforms_keys:
            self.pre_processing_operations.append(torchvision.transforms.Resize(transforms['target_size']))

        if 'grayscale' in transforms_keys and transforms['grayscale']:
            # num_output_channels = 3 because pre-trained models use 3 channels
            self.pre_processing_operations.append(torchvision.transforms.Grayscale(num_output_channels=3))

        self.pre_processing_operations.append(torchvision.transforms.ToTensor())

    def get_transforms(self):
        """Get the transformations to be applied to the images."""
        transforms = torchvision.transforms.Compose(self.pre_processing_operations)
        return transforms


class ImageDataAugmentation:
    """Defines the images data augmentation transformations."""

    def __init__(self, transforms: dict = {'rotate': 5}, pre_processing_operations=None):
        """
        Collect the transformations to be applied to the images.

        Order of operations is pre defined here.
        """
        self.pre_processing_operations = pre_processing_operations
        if pre_processing_operations is None:
            self.pre_processing_operations = []

        # Removing to_tensor transformation coming from pre processing. It must be done at the end.
        self.pre_processing_operations = [
            p for p in self.pre_processing_operations if not isinstance(p, torchvision.transforms.transforms.ToTensor)
        ]

        if 'rotate' in transforms.keys():
            self.pre_processing_operations.append(torchvision.transforms.RandomRotation(transforms['rotate']))

        self.pre_processing_operations.append(torchvision.transforms.ToTensor())

    def get_transforms(self):
        """Get the transformations to be applied to the images."""
        transforms = torchvision.transforms.Compose(self.pre_processing_operations)
        return transforms


class Module:
    """The base module class."""

    def __init__(self):
        """Initialize module."""
        raise NotImplementedError

    def forward(self):
        """Forward pass."""
        raise NotImplementedError

    def from_pretrained(self, load: Union[None, str, Dict] = None):
        """Load a module from a pre-trained state."""
        if load is None:
            pass
        elif isinstance(load, str):
            # load is a string so we assume it's a path to the saved state dict
            self.load_state_dict(torch.load(load))
        elif isinstance(load, dict):
            # load is a dict so we assume it's the state dict which we load directly
            self.load_state_dict(load)
        else:
            raise ValueError(f'input to from_pretrained should be None, str or dict, got {type(load)}')


class Classifier:
    """The base Classifier that all classifiers inheret from."""

    def __init__(self):
        """Initialize."""
        raise NotImplementedError

    def forward(self):
        """Forward pass."""
        raise NotImplementedError


class DocumentTextClassifier(nn.Module, Classifier):
    """Classifies a document based on the text on each page only."""

    def __init__(self, text_module: nn.Module, output_dim: int, dropout_rate: float = 0.0, **kwargs):
        """Initialize the classifier."""
        super().__init__()

        assert isinstance(text_module, Module)

        self.text_module = text_module
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self.fc_out = nn.Linear(text_module.n_features, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.FloatTensor]:
        """Forward pass."""
        encoded_text = self.text_module(input)
        text_features = encoded_text['features']
        # text_features = [batch, seq len, n text features]
        pooled_text_features = text_features.mean(dim=1)  # mean pool across sequence length
        # pooled_text_features = [batch, n text features]
        prediction = self.fc_out(self.dropout(pooled_text_features))
        # prediction = [batch, output dim]
        output = {'prediction': prediction}
        if 'attention' in encoded_text:
            output['attention'] = encoded_text['attention']
        return output


class DocumentImageClassifier(nn.Module, Classifier):
    """Classifies a document based on the image of the pages only."""

    def __init__(self, image_module: nn.Module, output_dim: int, dropout_rate: float = 0.0, **kwargs):
        """Initialize the classifier."""
        super().__init__()

        assert isinstance(image_module, Module)

        self.image_module = image_module
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self.fc_out = nn.Linear(image_module.n_features, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.FloatTensor]:
        """Forward pass."""
        encoded_image = self.image_module(input)
        image_features = encoded_image['features']
        # image_features = [batch, n image features]
        prediction = self.fc_out(self.dropout(image_features))
        # prediction = [batch, output dim]
        output = {'prediction': prediction}
        return output


class DocumentMultimodalClassifier(nn.Module, Classifier):
    """
    Model to classify document pages.

    It can take in consideration the combination of the document visual and text features.
    """

    def __init__(
        self,
        image_module: nn.Module,
        text_module: nn.Module,
        multimodal_module: nn.Module,
        output_dim: int,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__()

        assert isinstance(text_module, Module)
        assert isinstance(image_module, Module)
        assert isinstance(multimodal_module, Module)

        self.image_module = image_module  # input: images, output: image features
        self.text_module = text_module  # input: text, output: text features
        self.multimodal_module = multimodal_module  # input: (image feats, text feats), output: multimodal feats
        self.output_dim = output_dim

        self.fc_out = nn.Linear(multimodal_module.n_features, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.FloatTensor]:
        """Define the computation performed at every call."""
        encoded_image = self.image_module(input)
        image_features = encoded_image['features']
        # image_features = [batch, n image features]
        encoded_text = self.text_module(input)
        text_features = encoded_text['features']
        # text_features = [batch, seq length, n text features]
        pooled_text_features = text_features.mean(dim=1)  # mean pool across sequence length
        # text_features = [batch, n text features]
        multimodal_features = self.multimodal_module(image_features, pooled_text_features)['features']
        # prediction = [batch, n multimodal features]
        prediction = self.fc_out(self.dropout(multimodal_features))
        output = {'prediction': prediction}
        if 'attention' in encoded_text:
            output['attention'] = encoded_text['attention']
        return output


class NBOW(nn.Module, Module):
    """NBOW classification model."""

    def __init__(
        self,
        input_dim: int,
        emb_dim: int = 64,
        dropout_rate: float = 0.0,
        load: Union[None, str, dict] = None,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.n_features = emb_dim

        self.from_pretrained(load)

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.FloatTensor]:
        """Define the computation performed at every call."""
        text = input['text']
        # text = [batch, seq len]
        text_features = self.dropout(self.embedding(text))
        # text_features = [batch, seq len, emb dim]
        output = {'features': text_features}
        return output


class NBOWSelfAttention(nn.Module, Module):
    """NBOW classification model with multi-headed self attention."""

    def __init__(
        self,
        input_dim: int,
        emb_dim: int = 64,
        n_heads: int = 8,
        dropout_rate: float = 0.0,
        load: Union[None, str, dict] = None,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__()

        assert emb_dim % n_heads == 0, f'emb_dim ({emb_dim}) must be a multiple of n_heads ({n_heads})'

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.multihead_attention = nn.MultiheadAttention(emb_dim, n_heads)
        self.dropout = nn.Dropout(dropout_rate)

        self.n_features = emb_dim

        self.from_pretrained(load)

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.FloatTensor]:
        """Define the computation performed at every call."""
        text = input['text']
        # text = [batch, seq len]
        embeddings = self.dropout(self.embedding(text))
        # embeddings = [batch, seq len, emb dim]
        embeddings = embeddings.permute(1, 0, 2)
        text_features, attention = self.multihead_attention(embeddings, embeddings, embeddings)
        text_features = text_features.permute(1, 0, 2)
        # text_features = [batch, seq len, emb dim]
        # attention = [batch, seq len, seq len]
        output = {'features': text_features, 'attention': attention}
        return output


class LSTM(nn.Module, Module):
    """A long short-term memory (LSTM) model."""

    def __init__(
        self,
        input_dim: int,
        emb_dim: int = 64,
        hid_dim: int = 256,
        n_layers: int = 2,
        bidirectional: bool = True,
        dropout_rate: float = 0.0,
        load: Union[None, str, dict] = None,
        **kwargs,
    ):
        """Initialize LSTM model."""
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout_rate, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_rate)

        self.n_features = hid_dim * 2 if bidirectional else hid_dim

        self.from_pretrained(load)

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.FloatTensor]:
        """Forward pass."""
        text = input['text']
        # text = [batch size, seq len]
        embeddings = self.dropout(self.embedding(text))
        # embeddings = [batch size, seq len, emb dim]
        embeddings = embeddings.permute(1, 0, 2)
        # embeddings = [seq len, batch size, emb dim]
        text_features, _ = self.lstm(embeddings)
        # text_features = [seq len, batch size, hid dim * n directions]
        text_features = text_features.permute(1, 0, 2)
        # text_features = [batch size, seq len, hid dim * n directions]
        output = {'features': text_features}
        return output


class BERT(nn.Module, Module):
    """Wraps around pre-trained BERT-type models from the HuggingFace library."""

    def __init__(
        self,
        input_dim: int,
        name: str = 'bert-base-german-cased',
        freeze: bool = True,
        load: Union[None, str, dict] = None,
        **kwargs,
    ):
        """Initialize BERT model from the HuggingFace library."""
        super().__init__()

        self.input_dim = input_dim
        self.name = name
        self.freeze = freeze

        self.bert = transformers.AutoModel.from_pretrained(name)

        if freeze:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False

        if 'hidden_size' in self.bert.config.to_dict():
            self.n_features = self.bert.config.to_dict()['hidden_size']
        elif 'dim' in self.bert.config.to_dict():
            self.n_features = self.bert.config.to_dict()['dim']
        else:
            raise ValueError(f'Cannot find feature dim for model: {name}')

        self.from_pretrained(load)

    def get_max_length(self):
        """Get the maximum length of a sequence that can be passed to the BERT module."""
        return self.bert.config.max_position_embeddings

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.FloatTensor]:
        """Forward pass."""
        text = input['text']
        # text = [batch, seq len]
        bert_output = self.bert(text, output_attentions=True, return_dict=False)
        if len(bert_output) == 2:  # distill-bert models only output features and attention
            text_features, attentions = bert_output
        elif len(bert_output) == 3:  # standard bert models also output pooling layers, which we don't want
            text_features, _, attentions = bert_output
        else:
            raise ValueError('bert module returned incorrect output size!')
        # text_features = [batch size, seq len, hid dim]
        # attentions = a [batch size, n heads, seq len, seq len] sized tensor per layer in the bert model
        attention = attentions[-1].mean(dim=1)  # get the attention from the final layer and average across the heads
        # attention = [batch size, seq len, seq len]
        output = {'features': text_features, 'attention': attention}
        return output


def get_text_module(config: dict) -> torch.nn.Module:
    """Get the text module accordingly with the specifications."""
    assert 'name' in config, 'text_module config needs a `name`'
    assert 'input_dim' in config, 'text_module config needs an `input_dim`'
    module_name = config['name']
    if module_name == 'nbow':
        text_module = NBOW(**config)
    elif module_name == 'nbowselfattention':
        text_module = NBOWSelfAttention(**config)
    elif module_name == 'lstm':
        text_module = LSTM(**config)
    else:
        try:  # try and get a BERT-type model from the Transformers library
            if 'finbert' in module_name:  # ability to use `finbert` as an alias for `ProsusAI/finbert`
                config['name'] = 'ProsusAI/finbert'
            text_module = BERT(**config)
        except OSError:
            raise ValueError(f'{module_name} is not a valid text module!')

    return text_module


class VGG(nn.Module, Module):
    """VGG classifier."""

    def __init__(
        self,
        name: str = 'vgg11',
        pretrained: bool = True,
        freeze: bool = True,
        load: Union[None, str, dict] = None,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__()

        self.name = name
        self.pretrained = pretrained
        self.freeze = freeze
        self.vgg = getattr(torchvision.models, name)(pretrained=pretrained)
        del self.vgg.classifier  # remove classifier as not needed

        if freeze:
            for parameter in self.vgg.parameters():
                parameter.requires_grad = False

        self.n_features = 512 * 7 * 7

        self.from_pretrained(load)

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.FloatTensor]:
        """Define the computation performed at every call."""
        image = input['image']
        # image = [batch, channels, height, width]
        image_features = self.vgg.features(image)
        image_features = self.vgg.avgpool(image_features)
        image_features = image_features.view(-1, self.n_features)
        # image_features = [batch, 512*7*7 = 25088]
        output = {'features': image_features}
        return output


class EfficientNet(nn.Module, Module):
    """EfficientNet classifier."""

    def __init__(
        self,
        name: str = 'efficientnet_b0',
        pretrained: str = True,
        freeze: str = True,
        load: Union[None, str, dict] = None,
        **kwargs,
    ):
        """Initialize the model."""
        super().__init__()

        self.name = name
        self.pretrained = pretrained
        self.freeze = freeze
        self.efficientnet = timm.create_model(
            name, pretrained=pretrained, num_classes=0
        )  # don't want classifier at end of model

        if freeze:
            for parameter in self.efficientnet.parameters():
                parameter.requires_grad = False

        self.n_features = self.get_n_features()

        self.from_pretrained(load)

    def get_n_features(self):
        """Calculate number of output features based on given model."""
        x = torch.randn(1, 3, 100, 100)
        with torch.no_grad():
            y = self.efficientnet(x)
        return y.shape[-1]

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.FloatTensor]:
        """Forward pass."""
        image = input['image']
        # images = [batch, channels, height, width]
        image_features = self.efficientnet(image)
        # image_features = [batch, n_features]
        output = {'features': image_features}
        return output


def get_image_module(config: dict) -> torch.nn.Module:
    """Get the image module accordingly with the specifications."""
    module_name = config['name']
    if module_name.startswith('vgg'):
        image_module = VGG(**config)
    elif module_name.startswith('efficientnet'):
        image_module = EfficientNet(**config)
    else:
        raise ValueError(f'{module_name} not a valid image module!')

    return image_module


class MultimodalConcatenate(nn.Module, Module):
    """Defines how the image and text features are combined."""

    def __init__(
        self,
        n_image_features: int,
        n_text_features: int,
        hid_dim: int = 256,
        load: Union[None, str, dict] = None,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__()

        self.n_image_features = n_image_features
        self.n_text_features = n_text_features
        self.hid_dim = hid_dim

        self.fc1 = nn.Linear(n_image_features + n_text_features, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        # TODO: remove the below `if` check after the following are phased out:
        # classifier_modules.py
        # models_multimodal.py
        # multimodal_modules.py
        if 'output_dim' in kwargs:
            self.fc3 = nn.Linear(hid_dim, kwargs['output_dim'])

        self.n_features = hid_dim

        self.from_pretrained(load)

    def forward(
        self, image_features: torch.FloatTensor, text_features: torch.FloatTensor
    ) -> Dict[str, torch.FloatTensor]:
        """Define the computation performed at every call."""
        # image_features = [batch, n_image_features]
        # text_features = [batch, n_text_features]
        concat_features = torch.cat((image_features, text_features), dim=1)
        # concat_features = [batch, n_image_features + n_text_features]
        x = F.relu(self.fc1(concat_features))
        # x = [batch size, hid dim]
        x = F.relu(self.fc2(x))
        # x = [batch size, hid dim]
        # TODO: remove the below `if` check after the following are phased out:
        # classifier_modules.py
        # models_multimodal.py
        # multimodal_modules.py
        if hasattr(self, 'fc3'):
            x = F.relu(self.fc3(x))
        output = {'features': x}
        return output


def get_multimodal_module(config: dict) -> torch.nn.Module:
    """Get the multimodal module accordingly with the specifications."""
    module_name = config['name']
    if module_name == 'concatenate':
        multimodal_module = MultimodalConcatenate(**config)
    else:
        raise ValueError(f'{module_name} is not a valid multimodal module!')

    return multimodal_module


def build_text_vocab(projects: List[Project], tokenizer: Tokenizer, min_freq: int = 1, max_size: int = None) -> Vocab:
    """Build a vocabulary over the document text."""
    logger.info('building text vocab')

    counter = collections.Counter()

    # loop over projects and documents updating counter using the tokens in each document
    for project in projects:
        for document in project.documents:
            tokens = tokenizer.get_tokens(document.text)
            counter.update(tokens)

    assert len(counter) > 0, 'Did not find any tokens when building the text vocab!'

    # create the vocab
    text_vocab = Vocab(counter, min_freq, max_size)

    return text_vocab


def get_document_classifier(
    config: dict,
) -> Union[DocumentTextClassifier, DocumentImageClassifier, DocumentMultimodalClassifier]:
    """
    Get a DocumentClassifier (and encapsulated module(s)) from a config dict.

    If the config only has a image or text module, then the appropriate DocumentClassifier is returned,
    i.e. either a DocumentImageClassifier or DocumentTextClassifier. If the config has both modules then
    a DocumentMultimodalClassifier is returned.

    We also do some assertions to make sure no unnecessary modules are in the config dict, i.e if you want
    a DocumentTextClassifier then you should NOT have an `image_module` in your config dict.
    """
    # default assumption is that they are all none
    text_module = None
    image_module = None
    multimodal_module = None

    if 'image_module' in config and 'text_module' in config:
        assert (
            'multimodal_module' in config
        ), 'If you have both an image and text module then you also need a multimodal module!'

    assert 'output_dim' in config, 'No `output_dim` found in the document classifier\'s config!'

    # get text module if in config
    if 'text_module' in config:
        assert 'input_dim' in config['text_module'], 'Document classifier\'s `text_module` needs an `input_dim`!'
        text_module_config = config['text_module']
        text_module = get_text_module(text_module_config)
        config = deepcopy(config)
        del config['text_module']

    # get image module if in config
    if 'image_module' in config:
        image_module_config = config['image_module']
        image_module = get_image_module(image_module_config)
        config = deepcopy(config)
        del config['image_module']

    # get multimodal module if in config
    if 'multimodal_module' in config:
        # only get multimodal module if we have both an image and text module
        assert image_module is not None and text_module is not None
        multimodal_module_config = config['multimodal_module']
        multimodal_module_config['n_image_features'] = image_module.n_features
        multimodal_module_config['n_text_features'] = text_module.n_features
        multimodal_module = get_multimodal_module(multimodal_module_config)
        config = deepcopy(config)
        del config['multimodal_module']

    # if we have a text, image and multimodal module then we get a DocumentMultimodalClassifier
    # if not, then we get eiter a DocumentTextClassifier (if we only have an text module) or a DocumentImageClassifier
    # (if we only have an image module), we assert the modules not required are not in the config

    if text_module is not None and image_module is not None:
        document_classifier = DocumentMultimodalClassifier(image_module, text_module, multimodal_module, **config)
    elif text_module is not None:
        assert image_module is None
        document_classifier = DocumentTextClassifier(text_module, **config)
    elif image_module is not None:
        assert text_module is None
        document_classifier = DocumentImageClassifier(image_module, **config)
    else:
        raise ValueError("You did not pass an image or text module to your DocumentModel's config!")

    # need to ensure classifier starts in evaluation mode
    document_classifier.eval()

    return document_classifier


def get_optimizer(classifier: Classifier, config: dict) -> torch.optim.Optimizer:
    """Get an optimizer for a given `classifier` given a config."""
    logger.info('getting optimizer')

    # name of the optimizer, i.e. SGD, Adam, RMSprop
    optimizer_name = config['name']

    # need to remove name but not delete it from the actual config dictionary
    config = deepcopy(config)
    del config['name']

    # if optimizer is from transformers library, get from there
    # else get from pytorch optim
    if optimizer_name in {'Adafactor'}:
        optimizer = getattr(transformers.optimization, optimizer_name)(classifier.parameters(), **config)
        return optimizer
    else:
        optimizer = getattr(torch.optim, optimizer_name)(classifier.parameters(), **config)
        return optimizer


def get_document_classifier_data(
    documents: List[Document],
    tokenizer: Tokenizer,
    text_vocab: Vocab,
    category_vocab: Vocab,
    use_image: bool,
    use_text: bool,
    shuffle: bool,
    split_ratio: float,
    max_len: int,
):
    """
    Prepare the data necessary for the document classifier.

    For each document we split into pages and from each page we take:
      - the path to an image of the page
      - the tokenized and numericalized text on the page
      - the label (category) of the page
      - the id of the document
      - the page number
    """
    assert use_image or use_text, 'One of either `use_image` or `use_text` needs to be `True`!'

    data = []

    for document in documents:
        project_id = str(document.project.id)
        doc_info = get_document_classifier_examples(
            document, project_id, tokenizer, text_vocab, category_vocab, max_len, use_image, use_text
        )
        data.extend(zip(*doc_info))

    # shuffles the data
    if shuffle:
        random.shuffle(data)

    # creates a split if necessary
    n_split_examples = int(len(data) * split_ratio)
    # if we wanted at least some examples in the split, ensure we always have at least 1
    if split_ratio > 0.0:
        n_split_examples = max(1, n_split_examples)
    split_data = data[:n_split_examples]
    _data = data[n_split_examples:]

    return _data, split_data


def build_document_classifier_iterator(
    data: List,
    transforms: torchvision.transforms,
    text_vocab: Vocab,
    use_image: bool,
    use_text: bool,
    shuffle: bool,
    batch_size: int,
    device: torch.device = 'cpu',
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build the iterators for the data list."""
    logger.debug("build_document_classifier_iterator")

    def collate(batch, transforms) -> Dict[str, torch.LongTensor]:
        image_path, text, label, doc_id, page_num = zip(*batch)
        if use_image:
            # if we are using images, open as PIL images, apply transforms and place on GPU
            image = [Image.open(path) for path in image_path]
            image = torch.stack([transforms(img) for img in image], dim=0).to(device)
            image = image.to(device)
        else:
            # if not using images then just set to None
            image = None
        if use_text:
            # if we are using text, batch and pad the already tokenized and numericalized text and place on GPU
            text = torch.nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value=text_vocab.pad_idx)
            text = text.to(device)
        else:
            text = None
        # also place label on GPU
        # doc_id and page_num do not need to be placed on GPU
        label = torch.cat(label).to(device)
        doc_id = torch.cat(doc_id)
        page_num = torch.cat(page_num)
        # pack everything up in a batch dictionary
        batch = {'image': image, 'text': text, 'label': label, 'doc_id': doc_id, 'page_num': page_num}
        return batch

    # get the collate functions with the appropriate transforms
    data_collate = functools.partial(collate, transforms=transforms)

    # build the iterators
    iterator = DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collate)

    return iterator


def build_document_classifier_iterators(
    projects: List[Project],
    tokenizer: Tokenizer,
    eval_transforms: torchvision.transforms,
    train_transforms: torchvision.transforms,
    text_vocab: Vocab,
    category_vocab: Vocab,
    use_image: bool,
    use_text: bool,
    valid_ratio: float = 0.2,
    batch_size: int = 16,
    max_len: int = 50,
    device: torch.device = 'cpu',
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build the iterators for the document classifier."""
    assert use_image or use_text, 'One of either `use_image` or `use_text` needs to be `True`!'

    logger.info('building document classifier iterators')

    # get list documents per project
    train_documents = [project.documents for project in projects]
    test_documents = [project.test_documents for project in projects]

    # flatten
    train_documents = [doc for doc_list in train_documents for doc in doc_list]
    test_documents = [doc for doc_list in test_documents for doc in doc_list]

    # get data (list of examples) from Documents
    train_data, valid_data = get_document_classifier_data(
        train_documents,
        tokenizer,
        text_vocab,
        category_vocab,
        use_image,
        use_text,
        shuffle=True,
        split_ratio=valid_ratio,
        max_len=max_len,
    )

    test_data, _ = get_document_classifier_data(
        test_documents,
        tokenizer,
        text_vocab,
        category_vocab,
        use_image,
        use_text,
        shuffle=False,
        split_ratio=0,
        max_len=max_len,
    )

    logger.info(f'{len(train_data)} training examples')
    logger.info(f'{len(valid_data)} validation examples')
    logger.info(f'{len(test_data)} testing examples')

    train_iterator = build_document_classifier_iterator(
        train_data,
        train_transforms,
        text_vocab,
        use_image,
        use_text,
        shuffle=True,
        batch_size=batch_size,
        device=device,
    )
    valid_iterator = build_document_classifier_iterator(
        valid_data,
        eval_transforms,
        text_vocab,
        use_image,
        use_text,
        shuffle=False,
        batch_size=batch_size,
        device=device,
    )
    test_iterator = build_document_classifier_iterator(
        test_data, eval_transforms, text_vocab, use_image, use_text, shuffle=False, batch_size=batch_size, device=device
    )

    return train_iterator, valid_iterator, test_iterator


class DocumentModel(FallbackCategorizationModel):
    """A model that predicts a category for a given document."""

    def __init__(
        self,
        projects: Union[List[Project], None],
        tokenizer: Union[Tokenizer, str] = BPETokenizer(),
        image_preprocessing: Union[None, dict] = {'target_size': (1000, 1000), 'grayscale': True},
        image_augmentation: Union[None, dict] = {'rotate': 5},
        document_classifier_config: dict = {
            'image_module': {'name': 'efficientnet_b0'},
            'text_module': {'name': 'nbowselfattention'},
            'multimodal_module': {'name': 'concatenate'},
        },
        text_vocab: Union[None, Vocab] = None,
        category_vocab: Union[None, Vocab] = None,
        use_cuda: bool = True,
    ):
        """Initialize a DocumentModel."""
        # projects should be a list of at least 2 Projects or None
        # where None indicates no training will be done
        assert projects is None or (isinstance(projects, List) and all([isinstance(p, Project) for p in projects]))

        if projects is not None and len(projects) == 1:
            logger.info('You are only using 1 project for the document classification.')

        self.projects = projects
        self.tokenizer = tokenizer

        # if we are using an image module in our classifier then we need to set-up the
        # pre-processing and data augmentation for the images
        if 'image_module' in document_classifier_config:
            self.image_preprocessing = image_preprocessing
            self.image_augmentation = image_augmentation
            # get preprocessing
            preprocessing = ImagePreProcessing(transforms=image_preprocessing)
            preprocessing_ops = preprocessing.pre_processing_operations
            # get data augmentation
            augmentation = ImageDataAugmentation(
                transforms=image_augmentation, pre_processing_operations=preprocessing_ops
            )
            # evaluation transforms are just the preprocessing
            # training transforms are the preprocessing + augmentation
            self.eval_transforms = preprocessing.get_transforms()
            self.train_transforms = augmentation.get_transforms()
        else:
            # if not using an image module in our classifier then
            # our preprocessing and augmentation should be None
            assert (
                image_preprocessing is None and image_augmentation is None
            ), 'If not using an image module then preprocessing/augmentation must be None!'
            self.image_preprocessing = None
            self.image_augmentation = None
            self.eval_transforms = None
            self.train_transforms = None

        logger.info('setting up vocabs')

        # only build a text vocabulary if the classifier has a text module
        if 'text_module' in document_classifier_config:
            # ensure we have a tokenizer
            assert self.tokenizer is not None, 'If using a text module you must pass a Tokenizer!'

            if isinstance(self.tokenizer, str):
                self.tokenizer = get_tokenizer(tokenizer_name=self.tokenizer, projects=self.projects)

            if hasattr(tokenizer, 'vocab'):
                # some tokenizers already have a vocab so if they do we use that instead of building one
                self.text_vocab = tokenizer.vocab
                logger.info('Using tokenizer\'s vocab')
            elif text_vocab is None:
                # if our classifier has a text module we have a tokenizer that doesn't have a vocab
                # then we have to build a vocab from our projects using our tokenizer
                self.text_vocab = build_text_vocab(self.projects, self.tokenizer)
            else:
                self.text_vocab = text_vocab
                logger.info('Using provided text vocab')
            logger.info(f'Text vocab length: {len(self.text_vocab)}')
        else:
            # if the classifier doesn't have a text module then we shouldn't have a tokenizer
            # and the text vocab should be None
            assert tokenizer is None, 'If not using a text module then you should not pass a Tokenizer!'
            self.text_vocab = None

        # if we do not pass a category vocab then build one
        if category_vocab is None:
            self.category_vocab = build_category_vocab(self.projects)
        else:
            self.category_vocab = category_vocab
            logger.info('Using provided vocab')

        logger.info(f'Category vocab length: {len(self.category_vocab)}')
        logger.info(f'Category vocab counts: {self.category_vocab.counter}')

        logger.info('setting up document classifier')

        # set-up the document classifier
        # need to explicitly add input_dim and output_dim as they are calculated from the data
        if 'text_module' in document_classifier_config:
            document_classifier_config['text_module']['input_dim'] = len(self.text_vocab)
        document_classifier_config['output_dim'] = len(self.category_vocab)

        # store the classifier config file
        self.document_classifier_config = document_classifier_config

        # create document classifier from config
        self.classifier = get_document_classifier(document_classifier_config)

        self.device = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')

    def save(self, path: Union[None, str] = None, model_type: str = 'DocumentModel') -> None:
        """
        Save only the necessary parts of the model for extraction/inference.

        Saves:
        - tokenizer (needed to ensure we tokenize inference examples in the same way that they are trained)
        - transforms (to ensure we transform/pre-process images in the same way as training)
        - vocabs (to ensure the tokens/labels are mapped to the same integers as training)
        - configs (to ensure we load the same models used in training)
        - state_dicts (the classifier parameters achived through training)
        """
        # create dictionary to save all necessary model data
        data_to_save = {
            'tokenizer': self.tokenizer,
            'image_preprocessing': self.image_preprocessing,
            'image_augmentation': self.image_augmentation,
            'text_vocab': self.text_vocab,
            'category_vocab': self.category_vocab,
            'document_classifier_config': self.document_classifier_config,
            'document_classifier_state_dict': self.classifier.state_dict(),
            'model_type': model_type,
        }

        # Save only the necessary parts of the model for extraction/inference.
        # if no path is given then we use a default path and filename
        if path is None:
            path = os.path.join(self.projects[0].project_folder, 'models', f'{get_timestamp()}_{model_type}.pt')

        logger.info(f'Saving model of type {model_type} in {path}')

        # save all necessary model data
        torch.save(data_to_save, path)
        return path

    def get_accuracy(self, predictions: torch.FloatTensor, labels: torch.FloatTensor) -> torch.FloatTensor:
        """Calculate accuracy of predictions."""
        # predictions = [batch size, n classes]
        # labels = [batch size]
        batch_size, n_classes = predictions.shape
        # which class had the highest probability?
        top_predictions = predictions.argmax(dim=1)
        # top_predictions = [batch size]
        # how many of the highest probability predictions match the label?
        correct = top_predictions.eq(labels).sum()
        # divide by the batch size to get accuracy per batch
        accuracy = correct.float() / batch_size
        return accuracy

    def train(
        self, examples: DataLoader, classifier: Classifier, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> Tuple[List[float], List[float]]:
        """Perform one epoch of training."""
        classifier.train()
        losses = []
        accs = []
        for batch in tqdm.tqdm(examples, desc='Training'):
            predictions = classifier(batch)['prediction']
            loss = loss_fn(predictions, batch['label'])
            acc = self.get_accuracy(predictions, batch['label'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            accs.append(acc.item())
        return losses, accs

    @torch.no_grad()
    def evaluate(
        self, examples: DataLoader, classifier: Classifier, loss_fn: torch.nn.Module
    ) -> Tuple[List[float], List[float]]:
        """Evaluate the model, i.e. get loss and accuracy but do not update the model parameters."""
        classifier.eval()
        losses = []
        accs = []
        for batch in tqdm.tqdm(examples, desc='Evaluating'):
            predictions = classifier(batch)['prediction']
            loss = loss_fn(predictions, batch['label'])
            acc = self.get_accuracy(predictions, batch['label'])
            losses.append(loss.item())
            accs.append(acc.item())
        return losses, accs

    def fit_classifier(
        self,
        train_examples: DataLoader,
        valid_examples: DataLoader,
        test_examples: DataLoader,
        classifier: Classifier,
        n_epochs: int = 25,
        patience: int = 3,
        optimizer: dict = {'name': 'Adam'},
        lr_decay: float = 0.999,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Fits a classifier on given `train_examples` and evaluates on given `test_examples`.

        Uses performance on `valid_examples` to know when to stop training.
        Trains a model for n_epochs or until it runs out of patience (goes `patience` epochs without seeing the
        validation loss decrease), whichever comes first.
        """
        train_losses, train_accs = [], []
        valid_losses, valid_accs = [], []
        patience_counter = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = get_optimizer(classifier, optimizer)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=lr_decay)
        temp_dir = tempfile.gettempdir()
        temp_filename = os.path.join(temp_dir, f'temp_{uuid.uuid4().hex}.pt')
        logger.info('begin fitting')
        best_valid_loss = float('inf')
        for epoch in range(n_epochs):
            train_loss, train_acc = self.train(train_examples, classifier, loss_fn, optimizer)  # train epoch
            valid_loss, valid_acc = self.evaluate(valid_examples, classifier, loss_fn)  # validation epoch
            train_losses.extend(train_loss)  # keep track of all the losses/accs
            train_accs.extend(train_acc)
            valid_losses.extend(valid_loss)
            valid_accs.extend(valid_acc)
            logger.info(f'epoch: {epoch}')
            logger.info(f'train_loss: {np.mean(train_loss):.3f}, train_acc: {np.nanmean(train_acc):.3f}')
            logger.info(f'valid_loss: {np.mean(valid_loss):.3f}, valid_acc: {np.nanmean(valid_acc):.3f}')
            # use scheduler to reduce the lr
            scheduler.step(np.mean(valid_loss))
            # if we get the best validation loss so far
            # reset the patience counter, update the best loss value
            # and save the model parameters
            if np.mean(valid_loss) < best_valid_loss:
                patience_counter = 0
                best_valid_loss = np.mean(valid_loss)
                torch.save(classifier.state_dict(), temp_filename)
            # if we don't get the best validation loss so far
            # increase the patience counter
            else:
                patience_counter += 1
            # if we run out of patience, end fitting prematurely
            if patience_counter > patience:
                logger.info('lost patience!')
                break
        logger.info('training finished, begin testing')
        # load parameters that got us the best validation loss
        classifier.load_state_dict(torch.load(temp_filename))
        os.remove(temp_filename)
        # evaluate model over the test data
        test_losses, test_accs = self.evaluate(test_examples, classifier, loss_fn)
        logger.info(f'test_loss: {np.mean(test_losses):.3f}, test_acc: {np.nanmean(test_accs):.3f}')

        # bundle all the metrics together
        metrics = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'valid_losses': valid_losses,
            'valid_accs': valid_accs,
            'test_losses': test_losses,
            'test_accs': test_accs,
        }

        return classifier, metrics

    @torch.no_grad()
    def predict(self, examples: DataLoader, classifier: Classifier) -> Tuple[List[float], List[float]]:
        """Get predictions and true values of the input examples."""
        classifier.eval()
        predicted_classes = []
        actual_classes = []

        for batch in tqdm.tqdm(examples, desc='Evaluating'):
            predictions = classifier(batch)['prediction']
            predicted_classes.extend(predictions.argmax(dim=-1).cpu().numpy())
            actual_classes.extend(batch['label'].cpu().numpy())
        return predicted_classes, actual_classes

    def evaluate_classifier(self, test_examples, classifier, prediction_vocab):
        """Get the predicted and actual classes over the test set."""
        predicted_classes, actual_classes = self.predict(test_examples, classifier)

    @torch.no_grad()
    def predict_documents(self, examples: DataLoader, classifier: Classifier) -> Tuple[List[float], List[float]]:
        """Get predictions and true values of the input examples."""
        classifier.eval()
        actual_classes = []
        doc_ids = []
        raw_predictions = []

        for batch in tqdm.tqdm(examples, desc='Evaluating'):
            predictions = classifier(batch)['prediction']
            raw_predictions.extend(predictions)
            actual_classes.extend(batch['label'].cpu().numpy())
            doc_ids.extend(batch['doc_id'].cpu().numpy())
        return raw_predictions, actual_classes, doc_ids

    def evaluate_classifier_per_document(self, test_examples: List, classifier: Classifier, prediction_vocab: Vocab):
        """Get the predicted and actual classes over the test set."""
        predictions, actual_classes, doc_ids = self.predict_documents(test_examples, classifier)

        document_id_predictions = dict()
        document_id_actual = dict()

        for pred, actual, doc_id in zip(predictions, actual_classes, doc_ids):
            if str(doc_id) in document_id_predictions.keys():
                document_id_predictions[str(doc_id)].append(pred)
                document_id_actual[str(doc_id)] = actual
            else:
                document_id_predictions[str(doc_id)] = [pred]
                document_id_actual[str(doc_id)] = actual

        predicted_classes = []
        actual_classes = []

        for doc_id, actual in document_id_actual.items():
            page_predictions = torch.stack(document_id_predictions[doc_id])
            page_predictions = torch.softmax(page_predictions, dim=-1)  # [n pages, n classes]
            mean_page_prediction = page_predictions.mean(dim=0).cpu().numpy()
            predicted_class = mean_page_prediction.argmax()

            predicted_classes.append(predicted_class)
            actual_classes.append(actual)

        logger.info('\nResults per document\n')

    def build(self, document_training_config: dict = {}, **kwargs) -> Dict[str, List[float]]:
        """Trains the document classifier."""
        logger.info('getting document classifier iterators')

        # figure out if we need images and/or text depending on if the classifier
        # has an image and/or text module
        use_image = hasattr(self.classifier, 'image_module')
        use_text = hasattr(self.classifier, 'text_module')

        if hasattr(self.classifier, 'text_module') and isinstance(self.classifier.text_module, BERT):
            document_training_config['max_len'] = self.classifier.text_module.get_max_length()

        # get document classifier example iterators
        examples = build_document_classifier_iterators(
            self.projects,
            self.tokenizer,
            self.eval_transforms,
            self.train_transforms,
            self.text_vocab,
            self.category_vocab,
            use_image,
            use_text,
            **document_training_config,
            device=self.device,
        )

        train_examples, valid_examples, test_examples = examples

        # place document classifier on device
        self.classifier = self.classifier.to(self.device)

        logger.info('training label classifier')

        # fit the document classifier
        self.classifier, metrics = self.fit_classifier(
            train_examples, valid_examples, test_examples, self.classifier, **document_training_config
        )

        self.evaluate_classifier(test_examples, self.classifier, self.category_vocab)
        self.evaluate_classifier_per_document(test_examples, self.classifier, self.category_vocab)

        # put document classifier back on cpu to free up GPU memory
        self.classifier = self.classifier.to('cpu')

        return metrics

    @torch.no_grad()
    def extract(self, page_images, text, batch_size=2, *args, **kwargs) -> Tuple[Tuple[str, float], pd.DataFrame]:
        """
        Get the predicted category for a document.

        The document model can have as input the pages text and/or pages images.

        The output is a two element Tuple. The first elements contains the category
        (category template id or project id)
        with maximum confidence predicted by the model and the respective value of confidence (as a Tuple).
        The second element is a dataframe with all the categories and the respective confidence values.

        category | confidence
           A     |     x
           B     |     y

        In case the model wasn't trained to predict 'NO_LABEL' we can still have it in the output if
        the document falls
        in any of the following situations.

        The output prediction is 'NO_LABEL' if:

        - the number of images do not match the number pages text
        E.g.: document with 3 pages, 3 images and only 2 pages of text

        - empty page text. The output will be nan if it's the only page in the document.
        E.g.: blank page

        - the model itself predicts it
        E.g.: document different from the training data

        page_images: images of the document pages
        text: document text
        batch_size: number of samples for each prediction
        :return: tuple of (1) tuple of predicted category and respective confidence nad (2) predictions dataframe
        """
        # get device and place classifier on device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = self.classifier.to(device)

        # ensure backwards compatibility of models saved with project name instead of id.
        # Can be deleted when old models are adapted or not needed anymore
        try:
            # category_vocab saved with project id
            temp_categories = self.category_vocab.get_tokens()
            if 'NO_LABEL' in temp_categories:
                temp_categories.remove('NO_LABEL')
            _ = int(temp_categories[0])
            categories = self.category_vocab.get_tokens()
        except ValueError:
            # category_vocab saved with project name. Conversion is necessary for app compatibility
            categories = [str(NAMES_CONVERSION[name]) for name in self.category_vocab.get_tokens()]

        # split text into pages
        page_text = text.split('\f')

        # does our classifier use text and images?
        use_image = hasattr(self.classifier, 'image_module')
        use_text = hasattr(self.classifier, 'text_module')

        batch_image, batch_text = [], []
        predictions = []

        # prediction loop
        for i, (img, txt) in enumerate(zip(page_images, page_text)):
            if use_image:
                # if we are using images, open the image and perform preprocessing
                img = Image.open(img)
                img = self.eval_transforms(img)
                batch_image.append(img)
            if use_text:
                # if we are using text, tokenize and numericalize the text
                if isinstance(self.classifier.text_module, BERT):
                    max_length = self.classifier.text_module.get_max_length()
                else:
                    max_length = None
                tok = self.tokenizer.get_tokens(txt)[:max_length]
                # assert we have a valid token (e.g '\n\n\n' results in tok = [])
                if len(tok) <= 0:
                    logger.info(f'[WARNING] The token resultant from page {i} is empty. Page text: {txt}.')

                idx = [self.text_vocab.stoi(t) for t in tok]
                txt_coded = torch.LongTensor(idx)
                batch_text.append(txt_coded)
            # need to use an `or` here as we might not be using one of images or text
            if len(batch_image) >= batch_size or len(batch_text) >= batch_size or i == (len(page_images) - 1):
                # create the batch and get prediction per page
                batch = {}
                if use_image:
                    batch['image'] = torch.stack(batch_image).to(device)
                if use_text:
                    batch_text = torch.nn.utils.rnn.pad_sequence(
                        batch_text, batch_first=True, padding_value=self.text_vocab.pad_idx
                    )
                    batch['text'] = batch_text.to(device)

                if use_text and batch['text'].size()[1] == 0:
                    # There is no text in the batch. Text is empty (page token not valid).
                    # If using a Bert model, the prediction will fail. We skip the prediction and add nan instead.
                    prediction = torch.tensor([[np.nan for _ in range(len(self.category_vocab))]]).to(device)
                else:
                    prediction = self.classifier(batch)['prediction']

                predictions.extend(prediction)
                batch_image, batch_text = [], []

        # stack prediction per page, use softmax to convert to probability and average across
        predictions = torch.stack(predictions)  # [n pages, n classes]

        if predictions.shape != (len(page_images), len(self.category_vocab)):
            logger.error(
                f'[ERROR] Predictions shape {predictions.shape} different '
                f'than expected {(len(page_images), len(self.category_vocab))}'
            )
        predictions = torch.softmax(predictions, dim=-1).cpu().numpy()  # [n pages, n classes]

        # remove invalid pages
        predictions_filtered = [p for p in predictions if not all(np.isnan(x) for x in p)]

        mean_prediction = np.array(predictions_filtered).mean(axis=0)  # [n classes]

        # differences might happen due to floating points numerical errors
        if not math.isclose(sum(mean_prediction), 1.0, abs_tol=1e-4):
            logger.error(f'[ERROR] Sum of the predictions ({sum(mean_prediction)}) is not 1.0.')

        category_preds = dict()

        # store the prediction confidence per label
        for idx, label in enumerate(categories):
            category_preds[label] = mean_prediction[idx]

        # store prediction confidences in a df
        predictions_df = pd.DataFrame(
            data={'category': list(category_preds.keys()), 'confidence': list(category_preds.values())}
        )

        # which class did we predict?
        # what was the label of that class?
        # what was the confidence of that class?
        predicted_class = int(mean_prediction.argmax())
        predicted_label = int(categories[predicted_class])
        predicted_confidence = mean_prediction[predicted_class]

        return (predicted_label, predicted_confidence), predictions_df


def get_document_classifier_examples(
    document: Document,
    project_id: str,
    tokenizer: Tokenizer,
    text_vocab: Vocab,
    category_vocab: Vocab,
    max_len: int,
    use_image: bool,
    use_text: bool,
):
    """Get the per document examples for the document classifier."""
    document_image_paths = []
    document_tokens = []
    document_labels = []
    document_ids = []
    document_page_numbers = []

    # validate the data for the document
    if use_image:
        document.get_images()  # gets the images if they do not exist
        image_paths = [page.image_path for page in document.pages()]  # gets the paths to the images
        assert len(image_paths) > 0, f'No images found for document {document.id}'
    if use_text:
        page_texts = document.text.split('\f')
        assert len(page_texts) > 0, f'No text found for document {document.id}'

    # if only using image OR text then make the one not used a list of None
    if use_image and not use_text:
        page_texts = [None] * len(image_paths)
    elif use_text and not use_image:
        image_paths = [None] * len(page_texts)

    # check we have the same number of images and text pages
    # only useful when we have both an image and a text module
    assert len(image_paths) == len(
        page_texts
    ), f'No. of images ({len(image_paths)}) != No. of pages {len(page_texts)} for document {document.id}'

    for page_number, (image_path, page_text) in enumerate(zip(image_paths, page_texts)):
        if use_image:
            # if using an image module, store the path to the image
            document_image_paths.append(image_path)
        else:
            # if not using image module then don't need the image paths
            # so we just have a list of None to keep the lists the same length
            document_image_paths.append(None)
        if use_text:
            # if using a text module, tokenize the page, trim to max length and then numericalize
            page_tokens = tokenizer.get_tokens(page_text)[:max_len]
            document_tokens.append(torch.LongTensor([text_vocab.stoi(t) for t in page_tokens]))
        else:
            # if not using text module then don't need the tokens
            # so we just have a list of None to keep the lists the same length
            document_tokens.append(None)
        # append the label (category), the document's id number and the page number of each page
        document_labels.append(torch.LongTensor([category_vocab.stoi(project_id)]))
        document_ids.append(torch.LongTensor([document.id_]))
        document_page_numbers.append(torch.LongTensor([page_number]))

    return document_image_paths, document_tokens, document_labels, document_ids, document_page_numbers


def get_document_template_classifier_data(
    documents: List[Document],
    tokenizer: Tokenizer,
    text_vocab: Vocab,
    category_vocab: Vocab,
    use_image: bool,
    use_text: bool,
    shuffle: bool,
    split_ratio: float,
    max_len: int,
):
    """
    Prepare the data necessary for the document classifier.

    For each document we split into pages and from each page we take:
      - the path to an image of the page
      - the tokenized and numericalized text on the page
      - the label (category) of the page
      - the id of the document
      - the page number
    """
    assert use_image or use_text, 'One of either `use_image` or `use_text` needs to be `True`!'

    data = []

    for document in documents:
        # get document classification (defined by the category template)
        meta_data = document.project.meta_data
        template_id = [m['category_template'] for m in meta_data if m['id'] == document.id_]
        assert len(template_id) == 1
        template_id = str(template_id[0]) if template_id[0] else 'NO_LABEL'

        doc_info = get_document_classifier_examples(
            document, template_id, tokenizer, text_vocab, category_vocab, max_len, use_image, use_text
        )
        data.extend(zip(*doc_info))

    # shuffles the data
    if shuffle:
        random.shuffle(data)

    # creates a split if necessary
    n_split_examples = int(len(data) * split_ratio)
    # if we wanted at least some examples in the split, ensure we always have at least 1
    if split_ratio > 0.0:
        n_split_examples = max(1, n_split_examples)
    split_data = data[:n_split_examples]
    _data = data[n_split_examples:]

    return _data, split_data


def build_document_template_classifier_iterators(
    projects: List[Project],
    tokenizer: Tokenizer,
    eval_transforms: torchvision.transforms,
    train_transforms: torchvision.transforms,
    text_vocab: Vocab,
    category_vocab: Vocab,
    use_image: bool,
    use_text: bool,
    valid_ratio: float = 0.2,
    batch_size: int = 16,
    max_len: int = 50,
    device: torch.device = 'cpu',
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build the iterators for the document classifier."""
    assert use_image or use_text, 'One of either `use_image` or `use_text` needs to be `True`!'

    logger.info('building document classifier iterators')

    # get list documents per project
    train_documents = [project.documents for project in projects]
    test_documents = [project.test_documents for project in projects]

    # flatten
    train_documents = [doc for doc_list in train_documents for doc in doc_list]
    test_documents = [doc for doc_list in test_documents for doc in doc_list]

    # get data (list of examples) from Documents
    train_data, valid_data = get_document_template_classifier_data(
        train_documents,
        tokenizer,
        text_vocab,
        category_vocab,
        use_image,
        use_text,
        shuffle=True,
        split_ratio=valid_ratio,
        max_len=max_len,
    )

    test_data, _ = get_document_template_classifier_data(
        test_documents,
        tokenizer,
        text_vocab,
        category_vocab,
        use_image,
        use_text,
        shuffle=False,
        split_ratio=0,
        max_len=max_len,
    )

    logger.info(f'{len(train_data)} training examples')
    logger.info(f'{len(valid_data)} validation examples')
    logger.info(f'{len(test_data)} testing examples')

    train_iterator = build_document_classifier_iterator(
        train_data,
        train_transforms,
        text_vocab,
        use_image,
        use_text,
        shuffle=True,
        batch_size=batch_size,
        device=device,
    )
    valid_iterator = build_document_classifier_iterator(
        valid_data,
        eval_transforms,
        text_vocab,
        use_image,
        use_text,
        shuffle=False,
        batch_size=batch_size,
        device=device,
    )
    test_iterator = build_document_classifier_iterator(
        test_data, eval_transforms, text_vocab, use_image, use_text, shuffle=False, batch_size=batch_size, device=device
    )

    return train_iterator, valid_iterator, test_iterator


class CustomDocumentModel(DocumentModel):
    """A model that predicts a category for a given document trained with the category template."""

    def build(self, document_training_config: dict = {}) -> Dict[str, List[float]]:
        """Trains the document classifier."""
        logger.info('getting document classifier iterators')

        # figure out if we need images and/or text depending on if the classifier
        # has an image and/or text module
        use_image = hasattr(self.classifier, 'image_module')
        use_text = hasattr(self.classifier, 'text_module')

        if hasattr(self.classifier, 'text_module') and isinstance(self.classifier.text_module, BERT):
            document_training_config['max_len'] = self.classifier.text_module.get_max_length()

        # get document classifier example iterators
        examples = build_document_template_classifier_iterators(
            self.projects,
            self.tokenizer,
            self.eval_transforms,
            self.train_transforms,
            self.text_vocab,
            self.category_vocab,
            use_image,
            use_text,
            **document_training_config,
            device=self.device,
        )

        train_examples, valid_examples, test_examples = examples

        # place document classifier on device
        self.classifier = self.classifier.to(self.device)

        logger.info('training label classifier')

        # fit the document classifier
        self.classifier, metrics = self.fit_classifier(
            train_examples, valid_examples, test_examples, self.classifier, **document_training_config
        )

        self.evaluate_classifier(test_examples, self.classifier, self.category_vocab)
        self.evaluate_classifier_per_document(test_examples, self.classifier, self.category_vocab)

        # put document classifier back on cpu to free up GPU memory
        self.classifier = self.classifier.to('cpu')

        return metrics


def build_category_vocab(projects: List[Project]) -> Vocab:
    """Build a vocabulary over the categories of each document."""
    logger.info('building category vocab')

    counter = collections.Counter()

    # loop over projects, getting the id and name for each one
    counter.update([str(project.id) for project in projects])

    assert len(counter) > 0, 'Did not find any categories when building the category vocab!'

    # create the vocab
    category_vocab = Vocab(counter, min_freq=1, max_size=None, unk_token=None, pad_token=None)

    return category_vocab


def build_template_category_vocab(projects: List[Project]) -> Vocab:
    """Build a vocabulary over the categories of each annotation."""
    logger.info('building category vocab')

    counter = collections.Counter()

    # loop over projects, getting the id and name for each one
    for project in projects:
        prj_meta = project.meta_data
        temp = [
            str(met['category_template']) if met['category_template'] is not None else 'NO_LABEL' for met in prj_meta
        ]
        counter.update(temp)

    template_vocab = Vocab(
        counter, min_freq=1, max_size=None, unk_token=None, pad_token=None, special_tokens=['NO_LABEL']
    )

    assert len(template_vocab) > 0, 'Did not find any categories when building the category vocab!'

    # NO_LABEL should be label zero so we can avoid calculating accuracy over it later
    assert template_vocab.stoi('NO_LABEL') == 0

    return template_vocab


def create_transformations_dict(possible_transforms, args):
    """Create a dictionary with the transformations accordingly with input args."""
    input_dict = {}

    if isinstance(args, dict):
        for transform in possible_transforms:
            if args[transform] is not None:
                input_dict[transform] = args[transform]

    else:
        for transform in possible_transforms:
            if args.__dict__[transform] is not None:
                input_dict[transform] = args.__dict__[transform]

    if len(input_dict.keys()) == 0:
        return None

    return input_dict


def get_timestamp(format='%Y-%m-%d-%H-%M-%S') -> str:
    """
    Return formatted timestamp.

    :param format: Format of the timestamp (e.g. year-month-day-hour-min-sec)
    :return: Timestamp
    """
    now = datetime.datetime.now()
    timestamp = now.strftime(format)
    return timestamp


def build_category_document_model(
    project,
    category_model=None,
    document_classifier_config: Union[None, dict] = None,
    document_training_config: Union[None, dict] = None,
    img_args: Union[None, dict] = None,
    tokenizer_name: Union[None, str] = 'phrasematcher',
    output_dir=None,
    return_model=False,
    *args,
    **kwargs,
):
    """Build the document category classification model."""
    from konfuzio_sdk.trainer.tokenization import get_tokenizer

    projects = [project]
    category_model = category_model if category_model else CustomDocumentModel

    # Image transformations available
    possible_transformations_pre_processing = ['invert', 'target_size', 'grayscale']
    possible_transformations_data_augmentation = ['rotate']

    if document_classifier_config is None:
        # default model combines image and text features
        document_classifier_config = {
            'image_module': {'name': 'efficientnet_b0', 'pretrained': True, 'freeze': True},
            'text_module': {'name': 'nbowselfattention', 'emb_dim': 104},
            'multimodal_module': {'name': 'concatenate', 'hid_dim': 250},
        }

    pre_processing_transforms = None
    data_augmentation_transforms = None

    if 'image_module' in document_classifier_config.keys():
        # we only need the args for the image transformations if we are using image features
        if img_args is None:
            img_args = {'invert': False, 'target_size': (1000, 1000), 'grayscale': True, 'rotate': 5}

        pre_processing_transforms = create_transformations_dict(possible_transformations_pre_processing, img_args)
        data_augmentation_transforms = create_transformations_dict(possible_transformations_data_augmentation, img_args)

    tokenizer = None

    if 'text_module' in document_classifier_config.keys():
        # we only need the tokenizer if we are using text features
        tokenizer = get_tokenizer(tokenizer_name, project=project)

    category_vocab = build_template_category_vocab(projects)

    model = category_model(
        projects,
        tokenizer=tokenizer,
        image_preprocessing=pre_processing_transforms,
        image_augmentation=data_augmentation_transforms,
        document_classifier_config=document_classifier_config,
        category_vocab=category_vocab,
        *args,
        **kwargs,
    )

    if document_training_config is None:
        document_training_config = {
            'valid_ratio': 0.2,
            'batch_size': 6,
            'max_len': None,
            'n_epochs': 100,
            'patience': 3,
            'optimizer': {'name': 'Adam'},
        }

    model.build(document_training_config=document_training_config)

    model_type = 'DocumentModel'
    path = os.path.join(output_dir, f'{get_timestamp()}_{model_type}.pt')
    model_path = model.save(path=path)
    if return_model:
        return model_path, model
    return model_path
