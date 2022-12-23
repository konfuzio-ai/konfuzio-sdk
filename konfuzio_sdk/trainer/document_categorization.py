"""Implements a Categorization Model."""
import collections
import functools
import os
import math
import logging
import tempfile
import uuid
from copy import deepcopy
from typing import Union, List, Dict, Tuple, Optional
from warnings import warn
from io import BytesIO

import timm
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from konfuzio_sdk.data import Document, Page, Category
from konfuzio_sdk.evaluate import CategorizationEvaluation
from konfuzio_sdk.tokenizer.base import Vocab
from konfuzio_sdk.trainer.image import ImagePreProcessing, ImageDataAugmentation
from konfuzio_sdk.trainer.tokenization import Tokenizer
from konfuzio_sdk.utils import get_timestamp

logger = logging.getLogger(__name__)

warn('This module is WIP: https://gitlab.com/konfuzio/objectives/-/issues/9481', FutureWarning, stacklevel=2)


class FallbackCategorizationModel:
    """A simple, non-trainable model that predicts a Category for a given Document based on a predefined rule.

    It checks for whether the name of the Category is present in the input Document (case insensitive; also see
    Category.fallback_name). This can be an effective fallback logic to categorize Documents when no Categorization AI
    is available.
    """

    def __init__(self, categories: List[Category], *args, **kwargs):
        """Initialize FallbackCategorizationModel."""
        # Go through keyword arguments, and either save their values to our
        # instance, or raise an error.
        self.categories = categories
        self.name = self.__class__.__name__

        self.tokenizer = Tokenizer("dummy-tokenizer")
        self.evaluation = None

    def fit(self) -> None:
        """Use as placeholder Function."""
        raise NotImplementedError(
            f'{self} uses a fallback logic for categorizing Documents, and does not train a classifier.'
        )

    def save(self, output_dir: str, include_konfuzio=True):
        """Use as placeholder Function."""
        raise NotImplementedError(
            f'{self} uses a fallback logic for categorizing documents, this will not save model to disk.'
        )

    def evaluate(self, use_training_docs: bool = False) -> CategorizationEvaluation:
        """
        Evaluate the full Categorization pipeline on the pipeline's Test Documents.

        :param use_training_docs: Bool for whether to evaluate on the Training Documents instead of Test Documents.
        :return: Evaluation object.
        """
        eval_list = []
        if not use_training_docs:
            eval_docs = self.test_documents
        else:
            eval_docs = self.documents

        for document in eval_docs:
            predicted_doc = self.categorize(document=document, recategorize=True)
            eval_list.append((document, predicted_doc))

        self.evaluation = CategorizationEvaluation(self.categories, eval_list)

        return self.evaluation

    def _categorize_page(self, page: Page) -> Page:
        """Run categorization on a Page.

        :param page: Input Page
        :returns: The input Page with added Category information
        """
        for training_category in self.categories:
            if training_category.fallback_name in page.text.lower():
                page.category = training_category
                break
        if page.category is None:
            logger.warning(f'{self} could not find the Category of {page} by using the fallback categorization logic.')
        return page

    def categorize(self, document: Document, recategorize: bool = False, inplace: bool = False) -> Document:
        """Run categorization on a Document.

        :param document: Input Document
        :param recategorize: If the input Document is already categorized, the already present Category is used unless
        this flag is True

        :param inplace: Option to categorize the provided Document in place, which would assign the Category attribute
        :returns: Copy of the input Document with added categorization information
        """
        virtual_doc = deepcopy(document)
        if inplace:
            target_doc = document
        else:
            target_doc = virtual_doc
        if (document.category is not None) and (not recategorize):
            logger.info(
                f'In {document}, the Category was already specified as {document.category}, so it wasn\'t categorized '
                f'again. Please use recategorize=True to force running the Categorization AI again on this Document.'
            )
            return target_doc
        elif recategorize:
            virtual_doc._category = None
            for page in virtual_doc.pages():
                page.category = None

        # Categorize each Page of the Document.
        virtual_doc = self.tokenizer.tokenize(virtual_doc)
        for page in virtual_doc.pages():
            self._categorize_page(page)

        # Try to assign a Category to the Document itself.
        # If the Pages are differently categorized, the Document won't be assigned a Category at this stage.
        # The Document will have to be split at a later stage to find a consistent Category for each sub-Document.
        # Otherwise, the Category for each sub-Document (if any) will be corrected by the user.
        # virtual_doc = self._categorize_from_pages(virtual_doc)
        target_doc._category = virtual_doc._category
        for tpage, vpage in zip(target_doc.pages(), virtual_doc.pages()):
            tpage.category = vpage.category
        return target_doc


class ClassificationModule(nn.Module):
    """Define general functionality to work with nn.Module classes used for classification."""

    def _valid(self) -> None:
        """Validate architecture sizes."""
        raise NotImplementedError

    def _load_architecture(self) -> None:
        """Load NN architecture."""
        raise NotImplementedError

    def _define_features(self) -> None:
        """Define number of features as self.n_features: int."""
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


class TextClassificationModule(ClassificationModule):
    """Define general functionality to work with nn.Module classes used for text classification."""

    def __init__(
        self,
        input_dim: int,
        load: Union[None, str, dict] = None,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__()

        self.input_dim = input_dim

        for argk, argv in kwargs.items():
            setattr(self, argk, argv)

        self._valid()
        self._load_architecture()
        self._define_features()

        self.from_pretrained(load)

    def _output(self, text: torch.Tensor) -> List[torch.FloatTensor]:
        """Collect output of NN architecture."""
        raise NotImplementedError

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.FloatTensor]:
        """Define the computation performed at every call."""
        text = input['text']
        # text = [batch, seq len]
        outs = self._output(text)
        if len(outs) not in [1, 2]:
            raise TypeError(f"NN architecture of {self} returned {len(outs)} outputs, 1 or 2 expected.")
        output = {'features': outs[0]}
        if len(outs) == 2:
            output['attention'] = outs[1]
        return output


class NBOW(TextClassificationModule):
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
        super().__init__(input_dim=input_dim, emb_dim=emb_dim, dropout_rate=dropout_rate, load=load)

    def _valid(self) -> None:
        """Validate nothing as this NBOW implementation doesn't have constraints on input_dim or emb_dim."""
        pass

    def _load_architecture(self) -> None:
        """Load NN architecture."""
        self.embedding = nn.Embedding(self.input_dim, self.emb_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def _define_features(self) -> None:
        """Define the number of features as the embedding size."""
        self.n_features = self.emb_dim

    def _output(self, text: torch.Tensor) -> List[torch.FloatTensor]:
        """Collect output of the concatenation embedding -> dropout."""
        text_features = self.dropout(self.embedding(text))
        return [text_features]


class NBOWSelfAttention(TextClassificationModule):
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
        super().__init__(input_dim=input_dim, emb_dim=emb_dim, n_heads=n_heads, dropout_rate=dropout_rate, load=load)

    def _valid(self) -> None:
        """Check that the embedding size is a multiple of the number of heads."""
        assert (
            self.emb_dim % self.n_heads == 0
        ), f'emb_dim ({self.emb_dim}) must be a multiple of n_heads ({self.n_heads})'

    def _load_architecture(self) -> None:
        """Load NN architecture."""
        self.embedding = nn.Embedding(self.input_dim, self.emb_dim)
        self.multihead_attention = nn.MultiheadAttention(self.emb_dim, self.n_heads, batch_first=True)
        self.dropout = nn.Dropout(self.dropout_rate)

    def _define_features(self) -> None:
        """Define the number of features as the embedding size."""
        self.n_features = self.emb_dim

    def _output(self, text: torch.Tensor) -> List[torch.FloatTensor]:
        """Collect output of the multiple attention heads."""
        embeddings = self.dropout(self.embedding(text))
        # embeddings = [batch, seq len, emb dim]
        text_features, attention = self.multihead_attention(embeddings, embeddings, embeddings)
        return [text_features, attention]


class LSTM(TextClassificationModule):
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
        super().__init__(
            input_dim=input_dim,
            emb_dim=emb_dim,
            hid_dim=hid_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout_rate=dropout_rate,
            load=load,
        )

    def _valid(self) -> None:
        """Validate nothing as this LSTM implementation doesn't constrain input_dim, emb_dim, hid_dim or n_layers."""
        pass

    def _load_architecture(self) -> None:
        """Load NN architecture."""
        self.embedding = nn.Embedding(self.input_dim, self.emb_dim)
        self.lstm = nn.LSTM(
            self.emb_dim, self.hid_dim, self.n_layers, dropout=self.dropout_rate, bidirectional=self.bidirectional
        )
        self.dropout = nn.Dropout(self.dropout_rate)

    def _define_features(self) -> None:
        """If the architecture is bidirectional, the feature size is twice as large as the hidden layer size."""
        self.n_features = self.hid_dim * 2 if self.bidirectional else self.hid_dim

    def _output(self, text: torch.Tensor) -> List[torch.FloatTensor]:
        """Collect output of the LSTM model."""
        embeddings = self.dropout(self.embedding(text))
        # embeddings = [batch size, seq len, emb dim]
        embeddings = embeddings.permute(1, 0, 2)
        # embeddings = [seq len, batch size, emb dim]
        text_features, _ = self.lstm(embeddings)
        # text_features = [seq len, batch size, hid dim * n directions]
        text_features = text_features.permute(1, 0, 2)
        return [text_features]


class BERT(TextClassificationModule):
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
        super().__init__(input_dim=input_dim, name=name, freeze=freeze, load=load)

    def _valid(self) -> None:
        """Check that the specified HuggingFace model has a hidden_size key or a dim key in its configuration dict."""
        bert_config = self.bert.config.to_dict()
        if 'hidden_size' in bert_config:
            self._feature_size = 'hidden_size'
        if 'dim' in bert_config:
            self._feature_size = 'dim'
        else:
            raise ValueError(f'Cannot find feature dim for model: {self.name}')

    def _load_architecture(self) -> None:
        """Load NN architecture."""
        self.bert = transformers.AutoModel.from_pretrained(self.name)
        if self.freeze:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False

    def _define_features(self) -> None:
        """Define the feature size as the hidden layer size."""
        self.n_features = self.bert.config.to_dict()[self._feature_size]

    def get_max_length(self):
        """Get the maximum length of a sequence that can be passed to the BERT module."""
        return self.bert.config.max_position_embeddings

    def _output(self, text: torch.Tensor) -> List[torch.FloatTensor]:
        """Collect output of the HuggingFace BERT model."""
        bert_output = self.bert(text, output_attentions=True, return_dict=False)
        if len(bert_output) == 2:  # distill-bert models only output features and attention
            text_features, attentions = bert_output
        elif len(bert_output) == 3:  # standard bert models also output pooling layers, which we don't want
            text_features, _, attentions = bert_output
        else:
            raise ValueError(
                f'Unsupported output size for BERT module: returned {len(bert_output)} outputs, expected 2 or 3.'
            )
        # text_features = [batch size, seq len, hid dim]
        # attentions = a [batch size, n heads, seq len, seq len] sized tensor per layer in the bert model
        attention = attentions[-1].mean(dim=1)  # get the attention from the final layer and average across the heads
        # attention = [batch size, seq len, seq len]
        return [text_features, attention]


class DocumentClassifier(nn.Module):
    """Container for document classifiers."""

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.FloatTensor]:
        """Forward pass."""
        raise NotImplementedError


class DocumentTextClassifier(DocumentClassifier):
    """Classifies a document based on the text on each page only."""

    def __init__(self, text_module: TextClassificationModule, output_dim: int, dropout_rate: float = 0.0, **kwargs):
        """Initialize the classifier."""
        super().__init__()

        assert isinstance(text_module, TextClassificationModule)

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


class ImageClassificationModule(ClassificationModule):
    """Define general functionality to work with nn.Module classes used for image classification."""

    def __init__(
        self,
        name: str,
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

        self._valid()
        self._load_architecture()
        if freeze:
            self._freeze()

        self._define_features()

        self.from_pretrained(load)

    def _freeze(self) -> None:
        """Define how model weights are frozen."""
        raise NotImplementedError

    def _output(self, image: torch.Tensor) -> List[torch.FloatTensor]:
        """Collect output of NN architecture."""
        raise NotImplementedError

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.FloatTensor]:
        """Define the computation performed at every call."""
        image = input['image']
        # image = [batch, channels, height, width]
        image_features = self._output(image)
        # image_features = [batch, n_features]
        output = {'features': image_features}
        return output


class VGG(ImageClassificationModule):
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
        super().__init__(name, pretrained, freeze, load)

    def _valid(self) -> None:
        """No validations needed for this VGG implementation."""
        pass

    def _load_architecture(self) -> None:
        """Load NN architecture."""
        self.vgg = getattr(torchvision.models, self.name)(pretrained=self.pretrained)
        del self.vgg.classifier  # remove classifier as not needed

    def _freeze(self) -> None:
        """Define how model weights are frozen."""
        for parameter in self.vgg.parameters():
            parameter.requires_grad = False

    def _define_features(self) -> None:
        """VGG11 uses a 7x7x512 max pooling layer."""
        self.n_features = 512 * 7 * 7

    def _output(self, image: torch.Tensor) -> torch.FloatTensor:
        """Collect output of NN architecture."""
        image_features = self.vgg.features(image)
        image_features = self.vgg.avgpool(image_features)
        image_features = image_features.view(-1, self.n_features)
        return image_features


class EfficientNet(ImageClassificationModule):
    """EfficientNet classifier."""

    def __init__(
        self,
        name: str = 'efficientnet_b0',
        pretrained: bool = True,
        freeze: bool = True,
        load: Union[None, str, dict] = None,
        **kwargs,
    ):
        """Initialize the model."""
        super().__init__(name, pretrained, freeze, load)

    def _valid(self) -> None:
        """No validations needed for this EfficientNet implementation."""
        pass

    def _load_architecture(self) -> None:
        """Load NN architecture."""
        self.efficientnet = timm.create_model(
            self.name, pretrained=self.pretrained, num_classes=0
        )  # 0 classes as we don't want classifier at end of model

    def _freeze(self) -> None:
        """Define how model weights are frozen."""
        for parameter in self.efficientnet.parameters():
            parameter.requires_grad = False

    def get_n_features(self) -> int:
        """Calculate number of output features based on given model."""
        x = torch.randn(1, 3, 100, 100)
        with torch.no_grad():
            y = self.efficientnet(x)
        return y.shape[-1]

    def _define_features(self) -> None:
        """Depends on given EfficientNet model."""
        self.n_features = self.get_n_features()

    def _output(self, image: torch.Tensor) -> torch.FloatTensor:
        """Collect output of NN architecture."""
        image_features = self.efficientnet(image)
        return image_features


class DocumentImageClassifier(DocumentClassifier):
    """Classifies a document based on the image of the pages only."""

    def __init__(self, image_module: ImageClassificationModule, output_dim: int, dropout_rate: float = 0.0, **kwargs):
        """Initialize the classifier."""
        super().__init__()

        assert isinstance(image_module, ImageClassificationModule)

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


class MultimodalClassificationModule(ClassificationModule):
    """Define general functionality to work with nn.Module classes used for image and text classification."""

    def __init__(
        self,
        n_image_features: int,
        n_text_features: int,
        hid_dim: int = 256,
        output_dim: Optional[int] = None,
        load: Union[None, str, dict] = None,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__()

        self.n_image_features = n_image_features
        self.n_text_features = n_text_features
        self.hid_dim = hid_dim
        self.output_dim = output_dim

        self._valid()
        self._load_architecture()
        self._define_features()

        self.from_pretrained(load)

    def _output(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.FloatTensor:
        """Collect output of NN architecture."""
        raise NotImplementedError

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.FloatTensor]:
        """Define the computation performed at every call."""
        image_features = input['image_features']
        # image_features = [batch, n_image_features]
        text_features = input['text_features']
        # text_features = [batch, n_text_features]
        x = self._output(image_features, text_features)
        # x = [batch size, hid dim]
        output = {'features': x}
        return output


class MultimodalConcatenate(MultimodalClassificationModule):
    """Defines how the image and text features are combined."""

    def __init__(
        self,
        n_image_features: int,
        n_text_features: int,
        hid_dim: int = 256,
        output_dim: Optional[int] = None,
        load: Union[None, str, dict] = None,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__(n_image_features, n_text_features, hid_dim, output_dim, load)

    def _valid(self) -> None:
        """Validate nothing as this combination of text module and image module has no restrictions."""
        pass

    def _load_architecture(self) -> None:
        """Load NN architecture."""
        self.fc1 = nn.Linear(self.n_image_features + self.n_text_features, self.hid_dim)
        self.fc2 = nn.Linear(self.hid_dim, self.hid_dim)
        # TODO: remove the below `if` check after the following are phased out:
        # classifier_modules.py
        # models_multimodal.py
        # multimodal_modules.py
        if self.output_dim is not None:
            self.fc3 = nn.Linear(self.hid_dim, self.output_dim)

    def _define_features(self) -> None:
        """Define number of features as self.n_features: int."""
        self.n_features = self.hid_dim

    def _output(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Collect output of NN architecture."""
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
        return x


class DocumentMultimodalClassifier(DocumentClassifier):
    """Model to classify document pages.

    It can take in consideration the combination of the document visual and text features.
    """

    def __init__(
        self,
        image_module: ImageClassificationModule,
        text_module: TextClassificationModule,
        multimodal_module: MultimodalClassificationModule,
        output_dim: int,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__()

        assert isinstance(text_module, TextClassificationModule)
        assert isinstance(image_module, ImageClassificationModule)
        assert isinstance(multimodal_module, MultimodalClassificationModule)

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
        input = {'image_features': image_features, 'text_features': pooled_text_features}
        multimodal_features = self.multimodal_module(input)['features']
        # prediction = [batch, n multimodal features]
        prediction = self.fc_out(self.dropout(multimodal_features))
        output = {'prediction': prediction}
        if 'attention' in encoded_text:
            output['attention'] = encoded_text['attention']
        return output


def get_optimizer(classifier: DocumentClassifier, config: dict) -> torch.optim.Optimizer:
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


class CategorizationAI(FallbackCategorizationModel):
    """A trainable model that predicts a category for a given document."""

    def __init__(
        self,
        categories: List[Category],
        image_preprocessing=None,
        image_augmentation=None,
        use_cuda: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize a CategorizationAI."""
        super().__init__(categories, *args, **kwargs)
        self.pipeline_path = None
        self.documents = None
        self.test_documents = None

        self.tokenizer = None
        self.text_vocab = None
        self.category_vocab = None
        self.classifier = None

        # @TODO do not hardcode the config
        if image_augmentation is None:
            image_augmentation = {'rotate': 5}
        if image_preprocessing is None:
            image_preprocessing = {'target_size': (1000, 1000), 'grayscale': True}
        document_classifier_config: dict = {
            'image_module': {'name': 'efficientnet_b0'},
            'text_module': {'name': 'nbowselfattention'},
            'multimodal_module': {'name': 'concatenate'},
        }
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

        self.device = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')

    def save(self, path: Union[None, str] = None, model_type: str = 'CategorizationAI') -> str:
        """
        Save only the necessary parts of the model for extraction/inference.

        Saves:
        - tokenizer (needed to ensure we tokenize inference examples in the same way that they are trained)
        - transforms (to ensure we transform/pre-process images in the same way as training)
        - vocabs (to ensure the tokens/labels are mapped to the same integers as training)
        - configs (to ensure we load the same models used in training)
        - state_dicts (the classifier parameters achieved through training)
        """
        # create dictionary to save all necessary model data
        data_to_save = {
            'tokenizer': self.tokenizer,
            'image_preprocessing': self.image_preprocessing,
            'image_augmentation': self.image_augmentation,
            'text_vocab': self.text_vocab,
            'category_vocab': self.category_vocab,
            'classifier': self.classifier,
            'model_type': model_type,
        }

        # Save only the necessary parts of the model for extraction/inference.
        # if no path is given then we use a default path and filename
        if path is None:
            path = os.path.join(
                self.categories[0].project.project_folder, 'models', f'{get_timestamp()}_{model_type}.pt'
            )

        logger.info(f'Saving model of type {model_type} in {path}')

        # save all necessary model data
        torch.save(data_to_save, path)
        return path

    def build_template_category_vocab(self) -> Vocab:
        """Build a vocabulary over the Categories."""
        logger.info('building category vocab')

        counter = collections.Counter(NO_CATEGORY=0)

        counter.update([str(category.id_) for category in self.categories])

        template_vocab = Vocab(
            counter, min_freq=1, max_size=None, unk_token=None, pad_token=None, special_tokens=['NO_CATEGORY']
        )

        return template_vocab

    def build_text_vocab(self, min_freq: int = 1, max_size: int = None) -> Vocab:
        """Build a vocabulary over the document text."""
        logger.info('building text vocab')

        counter = collections.Counter()

        # loop over documents updating counter using the tokens in each document
        for document in self.documents:
            tokenized_document = self.tokenizer.tokenize(deepcopy(document))
            tokens = [span.offset_string for span in tokenized_document.spans()]
            counter.update(tokens)

        assert len(counter) > 0, 'Did not find any tokens when building the text vocab!'

        # create the vocab
        text_vocab = Vocab(counter, min_freq, max_size)

        return text_vocab

    def build_document_classifier_iterator(
        self,
        documents,
        transforms: torchvision.transforms,
        use_image: bool,
        use_text: bool,
        shuffle: bool,
        batch_size: int,
        max_len: int,
        device: torch.device = 'cpu',
    ) -> DataLoader:
        """
        Prepare the data necessary for the document classifier, and build the iterators for the data list.

        For each document we split into pages and from each page we take:
          - the path to an image of the page
          - the tokenized and numericalized text on the page
          - the label (category) of the page
          - the id of the document
          - the page number
        """
        logger.debug("build_document_classifier_iterator")

        # todo move this validation to the Categorization AI config
        assert use_image or use_text, 'One of either `use_image` or `use_text` needs to be `True`!'

        # get data (list of examples) from Documents
        data = []
        for document in documents:
            tokenized_doc = self.tokenizer.tokenize(deepcopy(document))
            tokenized_doc.status = document.status  # to allow to retrieve images from the original pages
            doc_info = tokenized_doc.get_document_classifier_examples(
                self.text_vocab, self.category_vocab, max_len, use_image, use_text
            )
            data.extend(zip(*doc_info))

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
                text = torch.nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value=self.text_vocab.pad_idx)
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

    def _train(
        self,
        examples: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> List[float]:
        """Perform one epoch of training."""
        self.classifier.train()
        losses = []
        for batch in tqdm.tqdm(examples, desc='Training'):
            predictions = self.classifier(batch)['prediction']
            loss = loss_fn(predictions, batch['label'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    def _fit_classifier(
        self,
        train_examples: DataLoader,
        n_epochs: int = 25,
        patience: int = 3,
        optimizer=None,
        lr_decay: float = 0.999,
        **kwargs,
    ) -> Tuple[DocumentClassifier, Dict[str, List[float]]]:
        """
        Fits a classifier on given `train_examples` and evaluates on given `test_examples`.

        Uses performance on `valid_examples` to know when to stop training.
        Trains a model for n_epochs or until it runs out of patience (goes `patience` epochs without seeing the
        validation loss decrease), whichever comes first.
        """
        if optimizer is None:
            optimizer = {'name': 'Adam'}
        train_losses = []
        patience_counter = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = get_optimizer(self.classifier, optimizer)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=lr_decay)
        temp_dir = tempfile.gettempdir()
        temp_filename = os.path.join(temp_dir, f'temp_{uuid.uuid4().hex}.pt')
        logger.info('begin fitting')
        best_valid_loss = float('inf')
        train_loss = float('inf')
        for epoch in range(n_epochs):
            train_loss = self._train(train_examples, loss_fn, optimizer)  # train epoch
            train_losses.extend(train_loss)  # keep track of all the losses/accs
            logger.info(f'epoch: {epoch}')
            logger.info(f'train_loss: {np.mean(train_loss):.3f}')
            # use scheduler to reduce the lr
            scheduler.step(np.mean(train_loss))
            # if we get the best validation loss so far
            # reset the patience counter, update the best loss value
            # and save the model parameters
            if np.mean(train_loss) < best_valid_loss:
                patience_counter = 0
                best_valid_loss = np.mean(train_loss)
                torch.save(self.classifier.state_dict(), temp_filename)
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
        self.classifier.load_state_dict(torch.load(temp_filename))
        os.remove(temp_filename)
        # evaluate model over the test data
        logger.info(f'test_loss: {np.mean(train_losses):.3f}, test_acc: {np.nanmean(train_loss):.3f}')

        training_metrics = {
            'train_losses': train_losses,
        }

        return self.classifier, training_metrics

    def fit(self, document_training_config=None, **kwargs) -> Dict[str, List[float]]:
        """Fit the CategorizationAI classifier."""
        if document_training_config is None:
            document_training_config = {}
        logger.info('getting document classifier iterators')

        # figure out if we need images and/or text depending on if the classifier
        # has an image and/or text module
        use_image = hasattr(self.classifier, 'image_module')
        use_text = hasattr(self.classifier, 'text_module')

        if hasattr(self.classifier, 'text_module') and isinstance(self.classifier.text_module, BERT):
            document_training_config['max_len'] = self.classifier.text_module.get_max_length()

        # @TODO move this validation to the Categorization AI configure step
        assert self.documents is not None, "Training documents need to be specified"
        assert self.test_documents is not None, "Test documents need to be specified"
        # get document classifier example iterators
        train_iterator = self.build_document_classifier_iterator(
            self.documents,
            self.train_transforms,
            use_image,
            use_text,
            shuffle=True,
            batch_size=document_training_config['batch_size'],
            max_len=document_training_config['max_len'],
            device=self.device,
        )
        # test_examples = build_document_classifier_iterator
        # examples = build_document_template_classifier_iterators(
        #     self.documents,
        #     self.test_documents,
        #     self.tokenizer,
        #     self.eval_transforms,
        #     self.train_transforms,
        #     self.text_vocab,
        #     self.category_vocab,
        #     use_image,
        #     use_text,
        #     **document_training_config,
        #     device=self.device,
        # )

        logger.info(f'{len(train_iterator)} training examples')
        # logger.info(f'{len(test_iterator)} testing examples')

        # place document classifier on device (this is a no-op if CPU was selected)
        self.classifier = self.classifier.to(self.device)

        logger.info('training label classifier')

        # fit the document classifier
        self.classifier, training_metrics = self._fit_classifier(train_iterator, **document_training_config)

        # put document classifier back on cpu to free up GPU memory (this is a no-op if CPU was already selected)
        self.classifier = self.classifier.to('cpu')

        return training_metrics

    @torch.no_grad()
    def _predict(self, page_images, text, batch_size=2, *args, **kwargs) -> Tuple[Tuple[int, float], pd.DataFrame]:
        """
        Get the predicted category for a document.

        The document model can have as input the pages text and/or pages images.

        The output is a two element Tuple. The first elements contains the category
        (category id or project id)
        with maximum confidence predicted by the model and the respective value of confidence (as a Tuple).
        The second element is a dataframe with all the categories and the respective confidence values.

        category | confidence
           A     |     x
           B     |     y

        In case the model wasn't trained to predict 'NO_CATEGORY' we can still have it in the output if
        the document falls in any of the following situations.

        The output prediction is 'NO_CATEGORY' if:

        - the number of images do not match the number pages text
        E.g.: document with 3 pages, 3 images and only 2 pages of text

        - empty page text. The output will be nan if it's the only page in the document.
        E.g.: blank page

        - the model itself predicts it
        E.g.: document different from the training data

        :param page_images: images of the document pages
        :param text: document text
        :param batch_size: number of samples for each prediction
        :return: tuple of (1) tuple of predicted category and respective confidence and (2) predictions dataframe
        """
        # get device and place classifier on device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = self.classifier.to(device)

        categories = self.category_vocab.get_tokens()

        # split text into pages
        page_text = text

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
                txt_coded = txt
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

    def _categorize_page(self, page: Page) -> Page:
        """Run categorization on a Page.

        :param page: Input Page
        :returns: The input Page with added categorization information
        """
        img_data = Image.open(page.image_path)
        buf = BytesIO()
        img_data.save(buf, format='PNG')
        docs_data_images = [buf]

        if isinstance(self.classifier.text_module, BERT):
            max_length = self.classifier.text_module.get_max_length()
        else:
            max_length = None
        self.text_vocab.numericalize(page, max_length)
        text_coded = [torch.LongTensor(page.text_encoded)]

        # todo optimize for gpu? self._predict can accept a batch of images/texts
        (predicted_category_id, predicted_confidence), _ = self._predict(page_images=docs_data_images, text=text_coded)

        if predicted_category_id == -1:
            # todo ensure that this never happens, then remove
            raise ValueError(f'{self} could not find the category of {page} by using the trained CategorizationAI.')

        for category in self.categories:
            if category.id_ == predicted_category_id:
                page.category = category
                break
        return page


# existent model classes
MODEL_CLASSES = {'CategorizationAI': CategorizationAI}

COMMON_PARAMETERS = ['tokenizer', 'text_vocab', 'model_type']

document_components = ['image_preprocessing', 'image_augmentation', 'category_vocab', 'classifier']

document_components.extend(COMMON_PARAMETERS)

# parameters that need to be saved with the model accordingly with the model type
MODEL_PARAMETERS_TO_SAVE = {'CategorizationAI': document_components}


def _load_categorization_model(path: str):
    """Load a Categorization model."""
    logger.info('loading model')

    # load model dict
    loaded_data = torch.load(path)

    if 'model_type' not in loaded_data.keys():
        model_type = path.split('_')[-1].split('.')[0]
    else:
        model_type = loaded_data['model_type']

    model_class = MODEL_CLASSES[model_type]
    model_args = MODEL_PARAMETERS_TO_SAVE[model_type]

    # Verify if loaded data has all necessary components
    if not all([arg in model_args for arg in loaded_data.keys()]):
        raise TypeError(f"Incomplete model parameters. Expected: {model_args}, Received: {list(loaded_data.keys())}")

    # create instance of the model class
    model = model_class(
        categories=None,
        image_preprocessing=loaded_data['image_preprocessing'],
        image_augmentation=loaded_data['image_augmentation'],
    )
    model.tokenizer = loaded_data['tokenizer']
    model.text_vocab = loaded_data['text_vocab']
    model.category_vocab = loaded_data['category_vocab']
    model.classifier = loaded_data['classifier']
    # need to ensure classifiers start in evaluation mode
    model.classifier.eval()

    return model


def load_categorization_model(pt_path: str, device: Optional[str] = 'cpu'):
    """
    Load a .pt (pytorch) file.

    :param pt_path: Path to the pytorch file.
    :param device: Device index or string to select. It’s a no-op if this argument is a negative integer or None.
    :raises FileNotFoundError: If the path is invalid.
    :raises OSError: When the data is corrupted or invalid and cannot be loaded.
    :raises TypeError: When the loaded pt file isn't recognized as a Konfuzio AI model.
    :return: Categorization AI model.
    """
    import dill

    # https://stackoverflow.com/a/43006034/5344492
    dill._dill._reverse_typemap['ClassType'] = type

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    with open(pt_path, 'rb') as f:  # todo check if we need to open 'rb' at all
        file_data = torch.load(pt_path, map_location=torch.device(device))

    if isinstance(file_data, dict):
        # verification of str in path can be removed after all models being updated with the model_type
        possible_names = list(MODEL_CLASSES.keys())
        if ('model_type' in file_data.keys() and file_data['model_type'] in possible_names) or any(
            [n in pt_path for n in possible_names]
        ):
            file_data = _load_categorization_model(pt_path)
        else:
            raise TypeError(f"Categorization Model type not recognized: {file_data['model_type']}")
    else:
        with open(pt_path, 'rb') as f:
            file_data = torch.load(f, map_location=torch.device(device))

    return file_data
