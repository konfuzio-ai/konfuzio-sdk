# ruff: noqa: A001, A002

"""Implements a Categorization Model."""

import abc
import collections
import functools
import io
import logging
import math
import os
import pathlib
import tempfile
import uuid
from copy import deepcopy
from enum import Enum
from inspect import signature
from typing import Dict, List, Optional, Tuple, Union

import lz4.frame
import numpy as np
import pandas as pd
import tqdm

from konfuzio_sdk.data import Category, CategoryAnnotation, Document, Page
from konfuzio_sdk.evaluate import CategorizationEvaluation
from konfuzio_sdk.extras import (
    DataLoader,
    FloatTensor,
    LongTensor,
    Module,
    Optimizer,
    Tensor,
    evaluate,
    timm,
    torch,
    torch_no_grad,
    torchvision,
    transformers,
)
from konfuzio_sdk.tokenizer.base import AbstractTokenizer, Vocab
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer
from konfuzio_sdk.trainer.base import BaseModel
from konfuzio_sdk.trainer.image import ImageDataAugmentation, ImagePreProcessing
from konfuzio_sdk.trainer.tokenization import TransformersTokenizer
from konfuzio_sdk.utils import get_timestamp

logger = logging.getLogger(__name__)


class AbstractCategorizationAI(BaseModel, metaclass=abc.ABCMeta):
    """Abstract definition of a CategorizationAI."""

    def __init__(self, categories: List[Category], *args, **kwargs):
        """Initialize AbstractCategorizationAI."""
        super().__init__()
        self.documents = None
        self.test_documents = None
        self.categories = categories
        self.project = None
        if categories is not None:
            self.project = categories[0].project
        self.evaluation = None

    def name_lower(self):
        """Convert class name to machine-readable name."""
        return f'{self.name.lower().strip()}'

    @property
    def temp_pkl_file_path(self) -> str:
        """
        Generate a path for temporary pickle file.

        :returns: A string with the path.
        """
        temp_pkl_file_path = os.path.join(
            self.project.project_folder,
            f'{get_timestamp()}_{self.project.id_}_{self.name_lower()}_tmp.pkl',
        )
        return temp_pkl_file_path

    @property
    def pkl_file_path(self) -> str:
        """
        Generate a path for a resulting pickle file.

        :returns: A string with the path.
        """
        pkl_file_path = os.path.join(
            self.project.project_folder,
            f'{get_timestamp()}_{self.project.id_}_{self.name_lower()}.pkl',
        )
        return pkl_file_path

    @abc.abstractmethod
    def fit(self) -> None:
        """Train the Categorization AI."""

    @abc.abstractmethod
    def save(self, output_dir: str, include_konfuzio=True):
        """Save the model to disk."""

    @abc.abstractmethod
    def _categorize_page(self, page: Page) -> Page:
        """Run categorization on a Page.

        :param page: Input Page
        :returns: The input Page with added CategoryAnnotation information
        """

    def categorize(self, document: Document, recategorize: bool = False, inplace: bool = False) -> Document:
        """Run categorization on a Document.

        :param document: Input Document
        :param recategorize: If the input Document is already categorized, the already present Category is used unless
        this flag is True

        :param inplace: Option to categorize the provided Document in place, which would assign the Category attribute
        :returns: Copy of the input Document with added CategoryAnnotation information
        """
        if inplace:
            virtual_doc = document
        else:
            virtual_doc = deepcopy(document)
        if (document.category not in [None, document.project.no_category]) and (not recategorize):
            logger.info(
                f"In {document}, the Category was already specified as {document.category}, so it wasn't categorized "
                f'again. Please use recategorize=True to force running the Categorization AI again on this Document.'
            )
            return virtual_doc

        # Categorize each Page of the Document.
        for page in virtual_doc.pages():
            self._categorize_page(page)

        return virtual_doc

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

    def check_is_ready(self):
        """
        Check if Categorization AI instance is ready for inference.

        It is assumed that the model is ready when there is at least one Category passed as the input.

        :raises AttributeError: When no Categories are passed into the model.
        """
        if not self.categories:
            raise AttributeError(f'{self} requires Categories.')

    @staticmethod
    def has_compatible_interface(other):
        """
        Validate that an instance of a Categorization AI implements the same interface as AbstractCategorizationAI.

        A Categorization AI should implement methods with the same signature as:
        - AbstractCategorizationAI.__init__
        - AbstractCategorizationAI.fit
        - AbstractCategorizationAI._categorize_page
        - AbstractCategorizationAI.check_is_ready

        :param other: An instance of a Categorization AI to compare with.
        """
        try:
            return (
                signature(other.__init__).parameters['categories'].annotation._name == 'List'
                and signature(other.__init__).parameters['categories'].annotation.__args__[0].__name__ == 'Category'
                and signature(other._categorize_page).parameters['page'].annotation.__name__ == 'Page'
                and signature(other._categorize_page).return_annotation.__name__ == 'Page'
                and signature(other.fit)
                and signature(other.check_is_ready)
            )
        except KeyError:
            return False
        except AttributeError:
            return False

    @staticmethod
    def load_model(pickle_path: str, device='cpu'):
        """
        Load the model and check if it has the interface compatible with the class.

        :param pickle_path: Path to the pickled model.
        :type pickle_path: str
        :raises FileNotFoundError: If the path is invalid.
        :raises OSError: When the data is corrupted or invalid and cannot be loaded.
        :raises TypeError: When the loaded pickle isn't recognized as a Konfuzio AI model.
        :return: Categorization AI model.
        """
        model = load_categorization_model(pickle_path, device)
        if not AbstractCategorizationAI.has_compatible_interface(model):
            raise TypeError(
                "Loaded model's interface is not compatible with any AIs. Please provide a model that has all the "
                'abstract methods implemented.'
            )
        return model


class NameBasedCategorizationAI(AbstractCategorizationAI):
    """A simple, non-trainable model that predicts a Category for a given Document based on a predefined rule.

    It checks for whether the name of the Category is present in the input Document (case insensitive; also see
    Category.fallback_name). This can be an effective fallback logic to categorize Documents when no Categorization AI
    is available.
    """

    def fit(self) -> None:
        """Use as placeholder Function because there's no classifier to be trainer."""
        raise NotImplementedError(
            f'{self} uses a fallback logic for categorizing Documents, and does not train a classifier.'
        )

    def save(self, output_dir: str, include_konfuzio=True):
        """Use as placeholder Function."""
        raise NotImplementedError(
            f'{self} uses a fallback logic for categorizing Documents, this will not save model to disk.'
        )

    def _categorize_page(self, page: Page) -> Page:
        """Run categorization on a Page.

        :param page: Input Page
        :returns: The input Page with added Category information
        """
        for training_category in self.categories:
            if training_category.fallback_name in page.text.lower():
                _ = CategoryAnnotation(category=training_category, confidence=1.0, page=page)
                break
        if page.category is None:
            logger.info(
                f'{self} could not find the Category of {page} by using the fallback categorization logic.'
                f'We will now apply the same Category of the first Page to this Page (if any).'
            )
            first_page = page.document.pages()[0]
            _ = CategoryAnnotation(category=first_page.category, confidence=1.0, page=page)
        return page


class AbstractCategorizationModel(Module, metaclass=abc.ABCMeta):
    """Define general functionality to work with nn.Module classes used for categorization."""

    @abc.abstractmethod
    def _valid(self) -> None:
        """Validate architecture sizes."""

    @abc.abstractmethod
    def _load_architecture(self) -> None:
        """Load NN architecture."""

    @abc.abstractmethod
    def _define_features(self) -> None:
        """Define number of features as `self.n_features: int`."""


class AbstractTextCategorizationModel(AbstractCategorizationModel, metaclass=abc.ABCMeta):
    """Define general functionality to work with nn.Module classes used for text categorization."""

    def __init__(
        self,
        input_dim: int = 0,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__()

        self.input_dim = input_dim
        self.bidirectional = None
        self.emb_dim = None

        for argk, argv in kwargs.items():
            setattr(self, argk, argv)

        self._valid()
        self._load_architecture()
        self.uses_attention = False
        self._define_features()

    @abc.abstractmethod
    def _output(self, text: Tensor) -> List[FloatTensor]:
        """Collect output of NN architecture."""

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, FloatTensor]:
        """Define the computation performed at every call."""
        text = input['text']
        # text = [batch, seq len]
        outs = self._output(text)
        if len(outs) not in [1, 2]:
            raise TypeError(f'NN architecture of {self} returned {len(outs)} outputs, 1 or 2 expected.')
        output = {'features': outs[0]}
        if len(outs) == 2:
            output['attention'] = outs[1]
        return output


class NBOW(AbstractTextCategorizationModel):
    """
    The neural bag-of-words (NBOW) model is the simplest of models, it passes each token through an embedding layer.

    As shown in the fastText paper (https://arxiv.org/abs/1607.01759) this model is still able to achieve comparable
    performance to some deep learning models whilst being considerably faster.

    One downside of this model is that tokens are embedded without regards to the surrounding context in which they
    appear, e.g. the embedding for “May” in the two sentences “May I speak to you?” and “I am leaving on the 1st of May”
    are identical, even though they have different semantics.

    :param emb_dim: The dimensions of the embedding vector.
    :param dropout_rate: The amount of dropout applied to the embedding vectors.
    """

    def __init__(
        self,
        input_dim: int,
        emb_dim: int = 64,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__(input_dim=input_dim, emb_dim=emb_dim, dropout_rate=dropout_rate)

    def _valid(self) -> None:
        """Validate nothing as this NBOW implementation doesn't have constraints on input_dim or emb_dim."""
        pass

    def _load_architecture(self) -> None:
        """Load NN architecture."""
        self.embedding = torch.nn.Embedding(self.input_dim, self.emb_dim)
        self.dropout = torch.nn.Dropout(self.dropout_rate)

    def _define_features(self) -> None:
        """Define the number of features as the embedding size."""
        self.n_features = self.emb_dim

    def _output(self, text: Tensor) -> List[FloatTensor]:
        """Collect output of the concatenation embedding -> dropout."""
        text_features = self.dropout(self.embedding(text))
        return [text_features]


class NBOWSelfAttention(AbstractTextCategorizationModel):
    """
    This is an NBOW model with a multi-headed self-attention layer, which is added after the embedding layer.

    See details at https://arxiv.org/abs/1706.03762.
    The self-attention layer effectively contextualizes the output as now each hidden state is calculated from the
    embedding vector of a token and the embedding vector of all other tokens within the sequence.

    :param emb_dim: The dimensions of the embedding vector.
    :param dropout_rate: The amount of dropout applied to the embedding vectors.
    :param n_heads: The number of attention heads to use in the multi-headed self-attention layer. Note that `n_heads`
    must be a factor of `emb_dim`, i.e. `emb_dim % n_heads == 0`.
    """

    def __init__(
        self,
        input_dim: int,
        emb_dim: int = 64,
        n_heads: int = 8,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__(input_dim=input_dim, emb_dim=emb_dim, n_heads=n_heads, dropout_rate=dropout_rate)
        self.uses_attention = True

    def _valid(self) -> None:
        """Check that the embedding size is a multiple of the number of heads."""
        if self.emb_dim % self.n_heads != 0:
            raise ValueError(f'emb_dim ({self.emb_dim}) must be a multiple of n_heads ({self.n_heads})')

    def _load_architecture(self) -> None:
        """Load NN architecture."""
        self.embedding = torch.nn.Embedding(self.input_dim, self.emb_dim)
        self.multihead_attention = torch.nn.MultiheadAttention(self.emb_dim, self.n_heads)
        self.dropout = torch.nn.Dropout(self.dropout_rate)

    def _define_features(self) -> None:
        """Define the number of features as the embedding size."""
        self.n_features = self.emb_dim

    def _output(self, text: Tensor) -> List[FloatTensor]:
        """Collect output of the multiple attention heads."""
        embeddings = self.dropout(self.embedding(text))
        # transposing so that the batch size is first. the result is embeddings = [batch, seq len, emb dim]
        # this step is needed to imitate batch_first=True argument in MultiheadAttention, since in torch==1.8.1 it is
        # not present.
        embeddings = embeddings.transpose(1, 0)
        text_features, attention = self.multihead_attention(embeddings, embeddings, embeddings)
        text_features = text_features.transpose(1, 0)
        return [text_features, attention]


class LSTM(AbstractTextCategorizationModel):
    """
    The LSTM (long short-term memory) is a variant of an RNN (recurrent neural network).

    It feeds the input tokens through an embedding layer and then processes them sequentially with the LSTM, outputting
    a hidden state for each token. If the LSTM is bidirectional then it trains a forward and backward LSTM per layer
    and concatenates the forward and backward hidden states for each token.

    :param emb_dim: The dimensions of the embedding vector.
    :param hid_dim: The dimensions of the hidden states.
    :param n_layers: How many LSTM layers to use.
    :param bidirectional: If the LSTM should be bidirectional.
    :param dropout_rate: The amount of dropout applied to the embedding vectors and between LSTM layers if
    `n_layers > 1`.
    """

    def __init__(
        self,
        input_dim: int,
        emb_dim: int = 64,
        hid_dim: int = 256,
        n_layers: int = 2,
        bidirectional: bool = True,
        dropout_rate: float = 0.0,
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
        )

    def _valid(self) -> None:
        """Validate nothing as this LSTM implementation doesn't constrain input_dim, emb_dim, hid_dim or n_layers."""
        pass

    def _load_architecture(self) -> None:
        """Load NN architecture."""
        self.embedding = torch.nn.Embedding(self.input_dim, self.emb_dim)
        self.lstm = torch.nn.LSTM(
            self.emb_dim, self.hid_dim, self.n_layers, dropout=self.dropout_rate, bidirectional=self.bidirectional
        )
        self.dropout = torch.nn.Dropout(self.dropout_rate)

    def _define_features(self) -> None:
        """If the architecture is bidirectional, the feature size is twice as large as the hidden layer size."""
        self.n_features = self.hid_dim * 2 if self.bidirectional else self.hid_dim

    def _output(self, text: Tensor) -> List[FloatTensor]:
        """Collect output of the LSTM model."""
        embeddings = self.dropout(self.embedding(text))
        # embeddings = [batch size, seq len, emb dim]
        embeddings = embeddings.permute(1, 0, 2)
        # embeddings = [seq len, batch size, emb dim]
        text_features, _ = self.lstm(embeddings)
        # text_features = [seq len, batch size, hid dim * n directions]
        text_features = text_features.permute(1, 0, 2)
        return [text_features]


class BERT(AbstractTextCategorizationModel):
    """
    Wraps around pre-trained BERT-type models from the HuggingFace library.

    BERT (bidirectional encoder representations from Transformers) is a family of large Transformer models. The
    available BERT variants are all pre-trained models provided by the transformers library. It is usually infeasible
    to train a BERT model from scratch due to the significant amount of computation required. However, the pre-trained
    models can be easily fine-tuned on desired data.

    The BERT variants, i.e. name arguments, that are covered by internal tests are:
        - `bert-base-german-cased`
        - `bert-base-german-dbmdz-cased`
        - `bert-base-german-dbmdz-uncased`
        - `distilbert-base-german-cased`

    In theory, all variants beginning with `bert-base-*` and `distilbert-*` should work out of the box. Other BERT
    variants come with no guarantees.

    :param name: The name of the pre-trained BERT variant to use.
    :param freeze: Should the BERT model be frozen, i.e. the pre-trained parameters are not updated.
    """

    def __init__(
        self,
        name: str = 'bert-base-german-cased',
        freeze: bool = False,
        **kwargs,
    ):
        """Initialize BERT model from the HuggingFace library."""
        super().__init__(name=name, freeze=freeze)
        self.uses_attention = True

    def _valid(self) -> None:
        """Check that the specified HuggingFace model has a hidden_size key or a dim key in its configuration dict."""
        pass

    def _load_architecture(self) -> None:
        """Load NN architecture."""
        try:
            self.bert = transformers.AutoModel.from_pretrained(self.name)
        except Exception:
            raise ValueError(f'Could not load Transformer model {self.name}.')
        if self.freeze:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False
        bert_config = self.bert.config.to_dict()
        if 'hidden_size' in bert_config:
            self._feature_size = 'hidden_size'
        elif 'dim' in bert_config:
            self._feature_size = 'dim'
        else:
            raise ValueError(f'Cannot find feature dim for model: {self.name}')

    def _define_features(self) -> None:
        """Define the feature size as the hidden layer size."""
        self.n_features = self.bert.config.to_dict()[self._feature_size]

    def get_max_length(self):
        """Get the maximum length of a sequence that can be passed to the BERT module."""
        return self.bert.config.max_position_embeddings

    def _output(self, text: Tensor) -> List[FloatTensor]:
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


class PageCategorizationModel(Module):
    """Container for Categorization Models."""

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, FloatTensor]:
        """Forward pass."""
        raise NotImplementedError


class PageTextCategorizationModel(PageCategorizationModel):
    """Container for Text Categorization Models."""

    def __init__(
        self, text_model: AbstractTextCategorizationModel, output_dim: int, dropout_rate: float = 0.0, **kwargs
    ):
        """Initialize the Model."""
        super().__init__()

        assert issubclass(text_model.__class__, AbstractTextCategorizationModel)

        self.text_model = text_model
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        # !TODO output_dim is not equal to the number of categories, this needs to be fixed !!
        self.fc_out = torch.nn.Linear(text_model.n_features, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, FloatTensor]:
        """Forward pass."""
        encoded_text = self.text_model(input)
        text_features = encoded_text['features']
        # text_features = [batch, seq len, n text features]
        # !TODO here features of the [CLS] token must be extracted and not an average pooling on all tokens !!
        pooled_text_features = text_features.mean(dim=1)  # mean pool across sequence length
        # pooled_text_features = [batch, n text features]
        prediction = self.fc_out(self.dropout(pooled_text_features))
        # prediction = [batch, output dim]
        output = {'prediction': prediction}
        if 'attention' in encoded_text:
            output['attention'] = encoded_text['attention']
        return output


class AbstractImageCategorizationModel(AbstractCategorizationModel, metaclass=abc.ABCMeta):
    """Define general functionality to work with nn.Module classes used for image categorization."""

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        freeze: bool = True,
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

    @abc.abstractmethod
    def _freeze(self) -> None:
        """Define how model weights are frozen."""

    @abc.abstractmethod
    def _output(self, image: Tensor) -> List[FloatTensor]:
        """Collect output of NN architecture."""

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, FloatTensor]:
        """Define the computation performed at every call."""
        image = input['image']
        # image = [batch, channels, height, width]
        image_features = self._output(image)
        # image_features = [batch, n_features]
        output = {'features': image_features}
        return output


class VGG(AbstractImageCategorizationModel):
    """
    The VGG family of models are image classification models designed for the ImageNet.

    They are usually used as a baseline in image classification tasks, however are considerably larger - in terms of
    the number of parameters - than modern architectures.

    Available variants are: `vgg11`, `vgg13`, `vgg16`, `vgg19`, `vgg11_bn`, `vgg13_bn`, `vgg16_bn`, `vgg19_bn`. The
    number generally indicates the number of layers in the model, higher does not always mean better. The `_bn` suffix
    means that the VGG model uses Batch Normalization layers, this generally leads to better results.

    The pre-trained weights are taken from the [torchvision](https://github.com/pytorch/vision) library and are weights
    from a model that has been trained as an image classifier on ImageNet. Ideally, this means the images should be
    3-channel color images that are at least 224x224 pixels and should be normalized.

    :param name: The name of the VGG variant to use
    :param pretrained: If pre-trained weights for the VGG variant should be used
    :param freeze: If the parameters of the VGG variant should be frozen
    """

    def __init__(
        self,
        name: str = 'vgg11',
        pretrained: bool = True,
        freeze: bool = True,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__(name, pretrained, freeze)

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

    def _output(self, image: Tensor) -> FloatTensor:
        """Collect output of NN architecture."""
        image_features = self.vgg.features(image)
        image_features = self.vgg.avgpool(image_features)
        image_features = image_features.view(-1, self.n_features)
        return image_features


class EfficientNet(AbstractImageCategorizationModel):
    """
    EfficientNet is a family of convolutional neural network based models that are designed to be more efficient.

    The efficiency comes in terms of the number of parameters and FLOPS, compared to previous computer vision models
    whilst maintaining equivalent image classification performance.

    Available variants are: `efficientnet_b0`, `efficientnet_b1`, ..., `efficienet_b7`. With `b0` having the least
    amount of parameters and `b7` having the most.

    The pre-trained weights are taken from the timm library and have been trained on ImageNet, thus the same tips,
    i.e. normalization, that apply to the VGG models also apply here.

    :param name: The name of the EfficientNet variant to use
    :param pretrained: If pre-trained weights for the EfficientNet variant should be used
    :param freeze: If the parameters of the EfficientNet variant should be frozen
    """

    def __init__(
        self,
        name: str = 'efficientnet_b0',
        pretrained: bool = True,
        freeze: bool = True,
        **kwargs,
    ):
        """Initialize the model."""
        super().__init__(name, pretrained, freeze)

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

    def _output(self, image: Tensor) -> FloatTensor:
        """Collect output of NN architecture."""
        image_features = self.efficientnet(image)
        return image_features


class PageImageCategorizationModel(PageCategorizationModel):
    """Container for Image Categorization Models."""

    def __init__(
        self, image_model: AbstractImageCategorizationModel, output_dim: int, dropout_rate: float = 0.0, **kwargs
    ):
        """Initialize the Model."""
        super().__init__()

        assert isinstance(image_model, AbstractImageCategorizationModel)

        self.image_model = image_model
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self.fc_out = torch.nn.Linear(image_model.n_features, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, FloatTensor]:
        """Forward pass."""
        encoded_image = self.image_model(input)
        image_features = encoded_image['features']
        # image_features = [batch, n image features]
        prediction = self.fc_out(self.dropout(image_features))
        # prediction = [batch, output dim]
        output = {'prediction': prediction}
        return output


class AbstractMultimodalCategorizationModel(AbstractCategorizationModel, metaclass=abc.ABCMeta):
    """Define general functionality to work with nn.Module classes used for image and text categorization."""

    def __init__(
        self,
        n_image_features: int,
        n_text_features: int,
        hid_dim: int = 256,
        output_dim: Optional[int] = None,
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

    @abc.abstractmethod
    def _output(self, image_features: Tensor, text_features: Tensor) -> FloatTensor:
        """Collect output of NN architecture."""

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, FloatTensor]:
        """Define the computation performed at every call."""
        image_features = input['image_features']
        # image_features = [batch, n_image_features]
        text_features = input['text_features']
        # text_features = [batch, n_text_features]
        x = self._output(image_features, text_features)
        # x = [batch size, hid dim]
        output = {'features': x}
        return output


class MultimodalConcatenate(AbstractMultimodalCategorizationModel):
    """Defines how the image and text features are combined in order to yield a categorization prediction."""

    def __init__(
        self,
        n_image_features: int,
        n_text_features: int,
        hid_dim: int = 256,
        output_dim: Optional[int] = None,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__(n_image_features, n_text_features, hid_dim, output_dim)

    def _valid(self) -> None:
        """Validate nothing as this combination of text module and image module has no restrictions."""
        pass

    def _load_architecture(self) -> None:
        """Load NN architecture."""
        self.fc1 = torch.nn.Linear(self.n_image_features + self.n_text_features, self.hid_dim)
        self.fc2 = torch.nn.Linear(self.hid_dim, self.hid_dim)
        if self.output_dim is not None:
            self.fc3 = torch.nn.Linear(self.hid_dim, self.output_dim)

    def _define_features(self) -> None:
        """Define number of features as self.n_features: int."""
        self.n_features = self.hid_dim

    def _output(self, image_features: Tensor, text_features: Tensor) -> Tensor:
        """Collect output of NN architecture."""
        concat_features = torch.cat((image_features, text_features), dim=1)
        # concat_features = [batch, n_image_features + n_text_features]
        x = torch.nn.functional.relu(self.fc1(concat_features))
        # x = [batch size, hid dim]
        x = torch.nn.functional.relu(self.fc2(x))
        # x = [batch size, hid dim]
        if hasattr(self, 'fc3'):
            x = torch.nn.functional.relu(self.fc3(x))
        return x


class PageMultimodalCategorizationModel(PageCategorizationModel):
    """
    Container for Text and Image Categorization Models.

    It can take in consideration the combination of the Document visual and text features.
    """

    def __init__(
        self,
        image_model: AbstractImageCategorizationModel,
        text_model: AbstractTextCategorizationModel,
        multimodal_model: AbstractMultimodalCategorizationModel,
        output_dim: int,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__()

        assert isinstance(text_model, AbstractTextCategorizationModel)
        assert isinstance(image_model, AbstractImageCategorizationModel)
        assert isinstance(multimodal_model, AbstractMultimodalCategorizationModel)

        self.image_model = image_model  # input: images, output: image features
        self.text_model = text_model  # input: text, output: text features
        self.multimodal_model = multimodal_model  # input: (image feats, text feats), output: multimodal feats
        self.output_dim = output_dim

        self.fc_out = torch.nn.Linear(multimodal_model.n_features, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, FloatTensor]:
        """Define the computation performed at every call."""
        encoded_image = self.image_model(input)
        image_features = encoded_image['features']
        # image_features = [batch, n image features]
        encoded_text = self.text_model(input)
        text_features = encoded_text['features']
        # text_features = [batch, seq length, n text features]
        pooled_text_features = text_features.mean(dim=1)  # mean pool across sequence length
        # text_features = [batch, n text features]
        input = {'image_features': image_features, 'text_features': pooled_text_features}
        multimodal_features = self.multimodal_model(input)['features']
        # prediction = [batch, n multimodal features]
        prediction = self.fc_out(self.dropout(multimodal_features))
        output = {'prediction': prediction}
        if 'attention' in encoded_text:
            output['attention'] = encoded_text['attention']
        return output


def get_optimizer(classifier: PageCategorizationModel, config: dict) -> Optimizer:
    """Get an optimizer for a given Model given a config."""
    logger.info('Getting optimizer')

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


class CategorizationAI(AbstractCategorizationAI):
    """A trainable AI that predicts a Category for each Page of a given Document."""

    def __init__(
        self,
        categories: List[Category],
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

        self.device = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')
        self.train_transforms = None

    @property
    def temp_pt_file_path(self) -> str:
        """
        Generate a path for s temporary model file in .pt format.

        :returns: A string with the path.
        """
        temp_pt_file_path = os.path.join(
            self.output_dir,
            f'{get_timestamp()}_{self.name_lower()}_{self.project.id_}_tmp.pt',
        )
        return temp_pt_file_path

    @property
    def compressed_file_path(self) -> str:
        """
        Generate a path for a resulting compressed file in .lz4 format.

        :returns: A string with the path.
        """
        output_dir = os.path.join(
            self.output_dir,
            f'{get_timestamp()}_{self.name_lower()}_{self.project.id_}.pt.lz4',
        )
        return output_dir

    def save(self, output_dir: Union[None, str] = None, reduce_weight: bool = True, **kwargs) -> str:
        """
        Save only the necessary parts of the model for extraction/inference.

        Saves:
        - tokenizer (needed to ensure we tokenize inference examples in the same way that they are trained)
        - transforms (to ensure we transform/pre-process images in the same way as training)
        - vocabs (to ensure the tokens/labels are mapped to the same integers as training)
        - configs (to ensure we load the same models used in training)
        - state_dicts (the classifier parameters achieved through training)

        Note: "path" is a deprecated parameter, "output_dir" is used for the sake of uniformity across all AIs.

        :param output_dir: A path to save the model to.
        :type output_dir: str
        :param reduce_weight: Reduces the weight of a model by removing Documents and reducing weight of a Tokenizer.
        :type reduce_weight: bool
        """
        if 'path' in kwargs:
            raise ValueError("'path' is a deprecated argument. Use 'output_dir' to specify the path to save the model.")
        if reduce_weight and self.tokenizer:
            self.reduce_model_weight()

        if not output_dir:
            self.output_dir = self.project.model_folder
        else:
            self.output_dir = output_dir

        # make sure output dir exists
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # temp_pt_file_path is needed to save an intermediate .pt file that later will be compressed and deleted.
        temp_pt_file_path = self.temp_pt_file_path
        compressed_file_path = self.compressed_file_path

        if self.categories and reduce_weight:
            self.categories[0].project.lose_weight()

        # create dictionary to save all necessary model data
        data_to_save = {
            'tokenizer': self.tokenizer,
            'image_preprocessing': self.image_preprocessing,
            'image_augmentation': self.image_augmentation,
            'text_vocab': self.text_vocab,
            'category_vocab': self.category_vocab,
            'classifier': self.classifier,
            'eval_transforms': self.eval_transforms,
            'train_transforms': self.train_transforms,
            'categories': self.categories,
            'model_type': 'CategorizationAI',
        }

        # Save only the necessary parts of the model for extraction/inference.
        # if no path is given then we use a default path and filename

        logger.info(f'Saving model of type Categorization AI in {compressed_file_path}')

        # save all necessary model data
        torch.save(data_to_save, temp_pt_file_path)
        with open(temp_pt_file_path, 'rb') as f_in:
            with open(compressed_file_path, 'wb') as f_out:
                compressed = lz4.frame.compress(f_in.read())
                f_out.write(compressed)
        self.pipeline_path = compressed_file_path
        os.remove(temp_pt_file_path)
        return self.pipeline_path

    def build_preprocessing_pipeline(self, use_image: bool, image_augmentation=None, image_preprocessing=None) -> None:
        """Set up the pre-processing and data augmentation when necessary."""
        # if we are using an image model in our classifier then we need to set up the
        # pre-processing and data augmentation for the images
        if use_image:
            if image_augmentation is None:
                image_augmentation = {'rotate': 5}
            if image_preprocessing is None:
                image_preprocessing = {'target_size': (1000, 1000), 'grayscale': True}
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

    def build_template_category_vocab(self) -> Vocab:
        """Build a vocabulary over the Categories."""
        logger.info('building category vocab')

        counter = collections.Counter()
        counter['0'] = 0  # add a 0 category for the NO_CATEGORY category

        counter.update([str(category.id_) for category in self.categories])

        template_vocab = Vocab(counter, min_freq=1, max_size=None, unk_token=None, pad_token=None, special_tokens=['0'])
        assert template_vocab.stoi('0') == 0, '0 category should be mapped to 0 index!'

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
        transforms,
        use_image: bool,
        use_text: bool,
        shuffle: bool,
        batch_size: int,
        max_len: int,
        device='cpu',
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
        logger.debug('build_document_classifier_iterator')

        # todo move this validation to the Categorization AI config
        assert use_image or use_text, 'One of either `use_image` or `use_text` needs to be `True`!'

        # get data (list of examples) from Documents
        data = []
        for document in documents:
            tokenized_doc = deepcopy(document)
            if self.tokenizer is not None:
                if not isinstance(self.tokenizer, TransformersTokenizer):
                    tokenized_doc = self.tokenizer.tokenize(tokenized_doc)
            tokenized_doc.status = document.status  # to allow to retrieve images from the original pages
            document_images = []
            document_tokens = []
            document_labels = []
            document_ids = []
            document_page_numbers = []
            if use_image:
                tokenized_doc.get_images()  # gets the images if they do not exist
                image_paths = [page.image_path for page in tokenized_doc.pages()]  # gets the paths to the images
                # @TODO move this validation to the Document class or the Page class
                assert len(image_paths) > 0, f'No images found for document {tokenized_doc.id_}'
                if not use_text:  # if only using images then make texts a list of None
                    page_texts = [None] * len(image_paths)
            if use_text:
                page_texts = tokenized_doc.text.split('\f')
                # @TODO move this validation to the Document class or the Page class
                assert len(page_texts) > 0, f'No text found for document {tokenized_doc.id_}'
                if not use_image:  # if only using text then make images used a list of None
                    image_paths = [None] * len(page_texts)

            # check we have the same number of images and text pages
            # only useful when we have both an image and a text module
            # @TODO move this validation to the Document class or the Page class
            assert len(image_paths) == len(
                page_texts
            ), f'No. of images ({len(image_paths)}) != No. of pages {len(page_texts)} for document {tokenized_doc.id_}'

            for page in tokenized_doc.pages():
                if use_image:
                    # if using an image module, store the path to the image
                    document_images.append(page.get_image())
                else:
                    # if not using image module then don't need the image paths
                    # so we just have a list of None to keep the lists the same length
                    document_images.append(None)
                if use_text:
                    if self.classifier.text_model.__class__.__name__ == 'BERT':
                        self.tokenizer = TransformersTokenizer(tokenizer_name=self.classifier.text_model.name)
                        page.text_encoded = self.tokenizer(page.text, max_length=max_len)['input_ids']
                        # for TransformersTokenizer you need to squeeze the first dimension since
                        # the output of the tokenizer has an extra dimension if return_tensors is set to 'pt'
                        document_tokens.append(torch.LongTensor(page.text_encoded).squeeze(0))
                    else:
                        # REPLACE page_tokens = tokenizer.get_tokens(page_text)[:max_len]
                        # page_encoded = [text_vocab.stoi(span.offset_string) for span in
                        # self.spans(start_offset=page.start_offset, end_offset=page.end_offset)]
                        # document_tokens.append(torch.LongTensor(page_encoded))
                        # if using a text module, tokenize the page, trim to max length and then numericalize
                        self.text_vocab.numericalize(page)
                        document_tokens.append(torch.LongTensor(page.text_encoded))
                else:
                    # if not using text module then don't need the tokens
                    # so we just have a list of None to keep the lists the same length
                    document_tokens.append(None)
                # get document classification (defined by the category template)
                category_id = str(tokenized_doc.category.id_)
                # append the classification (category), the document's id number and the page number of each page
                document_labels.append(torch.LongTensor([self.category_vocab.stoi(category_id)]))
                doc_id = tokenized_doc.id_ or tokenized_doc.copy_of_id
                document_ids.append(torch.LongTensor([doc_id]))
                document_page_numbers.append(torch.LongTensor([page.index]))
            doc_info = zip(document_images, document_tokens, document_labels, document_ids, document_page_numbers)
            data.extend(doc_info)

        def collate(batch, transforms) -> Dict[str, LongTensor]:
            image, text, label, doc_id, page_num = zip(*batch)
            if use_image:
                # if we are using images, they are already loaded as `PIL.Image`s, apply transforms and place on GPU
                image = torch.stack([transforms(img) for img in image], dim=0).to(device)
                image = image.to(device)
            else:
                # if not using images then just set to None
                image = None
            if use_text:
                # if we are using text, batch and pad the already tokenized and numericalized text and place on GPU
                if isinstance(self.tokenizer, TransformersTokenizer):
                    padding_value = self.classifier.text_model.bert.config.to_dict().get('pad_token_id', 0)
                else:
                    padding_value = self.text_vocab.pad_idx
                text = torch.nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value=padding_value)
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
        iterator = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collate)

        return iterator

    def _train(
        self,
        examples: DataLoader,
        loss_fn: Module,
        optimizer: Optimizer,
    ) -> List[float]:
        """Perform one epoch of training."""
        self.classifier.train()
        losses = []
        predictions_list = []
        ground_truth_list = []
        for batch in tqdm.tqdm(examples, desc='Training'):
            predictions = self.classifier(batch)['prediction']
            ground_truth = batch['label']
            predictions_list.extend(torch.argmax(predictions, axis=1).tolist())
            ground_truth_list.extend(ground_truth.tolist())
            # use "evaluate" to compute train_f1
            loss = loss_fn(predictions, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses, predictions_list, ground_truth_list

    def _fit_classifier(
        self,
        train_examples: DataLoader,
        n_epochs: int = 20,
        patience: int = 3,
        optimizer=None,
        lr_decay: float = 0.999,
        **kwargs,
    ) -> Tuple[PageCategorizationModel, Dict[str, List[float]]]:
        """
        Fits a classifier on given `train_examples` and evaluates on given `test_examples`.

        Uses performance on `valid_examples` to know when to stop training.
        Trains a model for n_epochs or until it runs out of patience (goes `patience` epochs without seeing the
        validation loss decrease), whichever comes first.
        """
        if optimizer is None:
            optimizer = {'name': 'Adam', 'lr': 1e-4}  # default learning rate of Adam is 1e-3
        train_losses = []
        patience_counter = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = get_optimizer(self.classifier, optimizer)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=lr_decay)
        temp_dir = tempfile.gettempdir()
        temp_filename = os.path.join(temp_dir, f'temp_{uuid.uuid4().hex}.pt')
        f1_metric = evaluate.load('f1')
        acc_metric = evaluate.load('accuracy')
        logger.info('Begin fitting')
        best_valid_loss = float('inf')
        train_loss = float('inf')
        for epoch in range(n_epochs):
            train_loss, predictions_list, ground_truth_list = self._train(
                train_examples, loss_fn, optimizer
            )  # train epoch
            f1_score = f1_metric.compute(predictions=predictions_list, references=ground_truth_list, average='macro')[
                'f1'
            ]  # compute f1 score
            acc_score = acc_metric.compute(predictions=predictions_list, references=ground_truth_list)[
                'accuracy'
            ]  # compute accuracy score
            train_losses.extend(train_loss)  # keep track of all the losses/accs
            logger.info(
                f'Epoch: {epoch + 1} | '
                f'Train Loss: {np.mean(train_loss):.3f} | '
                f'Train F1: {f1_score:.3f} | '
                f'Train Accuracy: {acc_score:.3f} |'
            )
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
        logger.info('Training finished. Loading best model')
        # load parameters that got us the best validation loss
        self.classifier.load_state_dict(torch.load(temp_filename))
        os.remove(temp_filename)

        training_metrics = {
            'train_losses': train_losses,
        }

        return self.classifier, training_metrics

    def fit(self, max_len: bool = None, batch_size: int = 1, **kwargs) -> Dict[str, List[float]]:
        """Fit the CategorizationAI classifier."""
        logger.info(
            f'Fitting Categorization AI classifier using: '
            f'\n\tmax_len: {max_len} '
            f'\n\tbatch_size: {batch_size}'
            f'\n\tkwargs: {kwargs}'
        )

        # figure out if we need images and/or text depending on if the classifier
        # has an image and/or text module
        use_image = hasattr(self.classifier, 'image_model')
        use_text = hasattr(self.classifier, 'text_model')

        if hasattr(self.classifier, 'text_model') and isinstance(self.classifier.text_model, BERT):
            max_len = self.classifier.text_model.get_max_length()

        assert self.documents is not None, 'Training documents need to be specified'
        assert self.test_documents is not None, 'Test documents need to be specified'
        # get document classifier example iterators
        train_iterator = self.build_document_classifier_iterator(
            self.documents,
            self.train_transforms,
            use_image,
            use_text,
            shuffle=True,
            batch_size=batch_size,
            max_len=max_len,
            device=self.device,
        )
        logger.info(f'{len(self.documents)} Training Documents')
        logger.info(f'{len(train_iterator.dataset)} Training Pages')
        # logger.info(f'{len(test_iterator)} testing examples')

        # place document classifier on device (this is a no-op if CPU was selected)
        self.classifier = self.classifier.to(self.device)

        logger.info('Training label classifier')

        # fit the document classifier
        self.classifier, training_metrics = self._fit_classifier(train_iterator, **kwargs)

        # put document classifier back on cpu to free up GPU memory (this is a no-op if CPU was already selected)
        self.classifier = self.classifier.to('cpu')

        return training_metrics

    def reduce_model_weight(self):
        """Reduce the size of the model by running lose_weight on the tokenizer."""
        if not isinstance(self.tokenizer, TransformersTokenizer):
            self.tokenizer.lose_weight()

    @torch_no_grad
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
        device = self.device
        self.classifier = self.classifier.to(device)

        categories = self.category_vocab.get_tokens()
        # split text into pages
        page_text = text

        # does our classifier use text and images?
        use_image = hasattr(self.classifier, 'image_model')
        use_text = hasattr(self.classifier, 'text_model')

        batch_image, batch_text = [], []
        predictions = []

        # prediction loop
        for i, (img, txt) in enumerate(zip(page_images, page_text)):
            if use_image:
                # if we are using images, open the image and perform preprocessing
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
                    if not isinstance(self.tokenizer, TransformersTokenizer):
                        padding_value = self.text_vocab.pad_idx
                    else:
                        padding_value = self.classifier.text_model.bert.config.to_dict().get('pad_token_id', 0)
                    batch_text = torch.nn.utils.rnn.pad_sequence(
                        batch_text, batch_first=True, padding_value=padding_value
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

        category_preds = {}

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
        docs_data_images = [None]
        use_image = hasattr(self.classifier, 'image_model')
        if use_image:
            page.get_image()
            docs_data_images = [page.image]

        use_text = hasattr(self.classifier, 'text_model')
        text_coded = [None]
        if use_text:
            if isinstance(self.classifier.text_model, BERT):
                max_length = self.classifier.text_model.get_max_length()
                page.text_encoded = self.tokenizer(page.text, max_length=max_length)['input_ids']
                text_coded = [torch.LongTensor(page.text_encoded).squeeze(0)]
            else:
                if not page.spans():
                    self.tokenizer.tokenize(page.document)
                max_length = None
                self.text_vocab.numericalize(page, max_length)
                text_coded = [torch.LongTensor(page.text_encoded)]

        (predicted_category_id, predicted_confidence), _ = self._predict(page_images=docs_data_images, text=text_coded)

        for category in self.categories:
            if category.id_ == predicted_category_id:
                _ = CategoryAnnotation(category=category, confidence=predicted_confidence, page=page)
                break
        return page


class ImageModel(Enum):
    """
    We currently have two image modules available (VGG and EfficientNet), each have several variants.

    The image models each have their classification heads removed and generally, they return the output of the final
    pooling layer within the model which has been flattened to a `[batch_size, n_features]` tensor, where `n_features`
    is an attribute of the model.
    """

    VGG11 = 'vgg11'
    VGG13 = 'vgg13'
    VGG16 = 'vgg16'
    VGG19 = 'vgg19'
    EfficientNetB0 = 'efficientnet_b0'
    EfficientNetB1 = 'efficientnet_b1'
    EfficientNetB2 = 'efficientnet_b2'
    EfficientNetB3 = 'efficientnet_b3'
    EfficientNetB4 = 'efficientnet_b4'
    EfficientNetB5 = 'efficientnet_b5'
    EfficientNetB6 = 'efficientnet_b6'
    EfficientNetB7 = 'efficientnet_b7'
    EfficientNetB8 = 'efficientnet_b8'


class TextModel(Enum):
    """
    There are currently four text modules available (NBOW, NBOW Self Attention, LSTM, and BERT).

    Each module takes a sequence of tokens as input and outputs a sequence of "hidden states", i.e. one vector per
    input token. The size of each of the hidden states can be found with the module's `n_features` parameter.
    """

    NBOW = 'nbow'
    NBOWSelfAttention = 'nbowselfattention'
    LSTM = 'lstm'
    BERT = 'bert'


class Optimizer(Enum):
    """SGD and Adam Optimizers."""

    SGD = 'SGD'
    Adam = 'Adam'


def build_categorization_ai_pipeline(
    categories: List[Category],
    documents: List[Document],
    test_documents: List[Document],
    tokenizer: Optional[AbstractTokenizer] = None,
    image_model_name: Optional[ImageModel] = None,
    text_model_name: Optional[TextModel] = TextModel.NBOW,
    **kwargs,
) -> CategorizationAI:
    """

    Build a Categorization AI neural network by choosing an ImageModel and a TextModel.

    See an in-depth tutorial at https://dev.konfuzio.com/sdk/tutorials/data_validation/index.html
    """
    logger.info(
        f'Building categorization AI pipeline using: \
                \n\timage_model_name: {image_model_name} \
                \n\ttext_model_name: {text_model_name}'
    )
    # Configure Categories, with training and test Documents for the Categorization AI
    categorization_pipeline = CategorizationAI(categories)
    categorization_pipeline.documents = documents
    categorization_pipeline.test_documents = test_documents
    # Configure pipeline with the tokenizer, text vocab, and category vocab
    if tokenizer is None:
        tokenizer = WhitespaceTokenizer()
    categorization_pipeline.tokenizer = tokenizer
    categorization_pipeline.category_vocab = categorization_pipeline.build_template_category_vocab()
    # Configure image and text models
    if image_model_name is not None:
        if isinstance(image_model_name, str):
            try:
                image_model = next(model for model in ImageModel if model.value == image_model_name)
            except StopIteration:
                raise ValueError(f'{image_model} not found. Provide an existing name for the image model.')
        else:
            image_model = image_model_name
        image_model_class = None
        if 'efficientnet' in image_model.value:
            image_model_class = EfficientNet
        elif 'vgg' in image_model.value:
            image_model_class = VGG
        # Configure image model
        image_model = image_model_class(name=image_model.value)
    if text_model_name is not None:
        if isinstance(text_model_name, str):
            try:
                text_model = next(model for model in TextModel if model.value in text_model_name)
            except StopIteration:
                # use BERT as a default model if the text_model_name is not found in TextModel enums
                # if text_model_name is not a supported Transformer model
                # ValueError from within BERT or TransformersTokenizer will be raised
                text_model = TextModel.BERT
        else:
            text_model = text_model_name
            text_model_name = text_model.name
        text_model_class_mapping = {
            TextModel.NBOW: NBOW,
            TextModel.NBOWSelfAttention: NBOWSelfAttention,
            TextModel.LSTM: LSTM,
            TextModel.BERT: BERT,
        }
        text_model_class = text_model_class_mapping[text_model]
        # Configure text model
        # Check if the text_model_class is BERT
        if text_model_class.__name__ == 'BERT':
            text_model = text_model_class(name=text_model_name)
        else:
            categorization_pipeline.text_vocab = categorization_pipeline.build_text_vocab()
            text_model = text_model_class(input_dim=len(categorization_pipeline.text_vocab))
    # Configure the classifier (whether it predicts using only the image of the Page,
    # or only the text, or a MLP to concatenate both predictions)
    if image_model_name is None:
        categorization_pipeline.classifier = PageTextCategorizationModel(
            text_model=text_model,
            output_dim=len(categorization_pipeline.category_vocab),
        )
        categorization_pipeline.build_preprocessing_pipeline(use_image=False)
    elif text_model_name is None:
        categorization_pipeline.classifier = PageImageCategorizationModel(
            image_model=image_model,
            output_dim=len(categorization_pipeline.category_vocab),
        )
        categorization_pipeline.build_preprocessing_pipeline(use_image=True)
    else:
        # If both image and text classification are chosen, create also a concatenation model
        multimodal_model = MultimodalConcatenate(
            n_image_features=image_model.n_features,
            n_text_features=text_model.n_features,
        )
        # Provide the classifier with image model, text model, and concatenation model
        categorization_pipeline.classifier = PageMultimodalCategorizationModel(
            image_model=image_model,
            text_model=text_model,
            multimodal_model=multimodal_model,
            output_dim=len(categorization_pipeline.category_vocab),
        )
        categorization_pipeline.build_preprocessing_pipeline(use_image=True)
    # need to ensure classifier starts in evaluation mode
    categorization_pipeline.classifier.eval()

    return categorization_pipeline


COMMON_PARAMETERS = ['tokenizer', 'text_vocab', 'model_type']

document_components = [
    'image_preprocessing',
    'image_augmentation',
    'category_vocab',
    'classifier',
    'eval_transforms',
    'train_transforms',
    'categories',
]

document_components.extend(COMMON_PARAMETERS)

# parameters that need to be saved with the model accordingly with the model type
MODEL_PARAMETERS_TO_SAVE = {'CategorizationAI': document_components}


def _load_categorization_model(path: str):
    """Load a Categorization model."""
    logger.info('loading model')

    # load model dict
    loaded_data = torch.load(path)

    model_type = 'CategorizationAI'
    # if 'model_type' not in loaded_data.keys():
    #    model_type = path.split('_')[-1].split('.')[0]
    # else:
    #    model_type = loaded_data['model_type']

    model_class = CategorizationAI
    model_args = MODEL_PARAMETERS_TO_SAVE[model_type]

    # Non-backwards compatible components to skip on the verification for loaded data.
    optional_components = [
        'categories',
    ]

    # Verify if loaded data has all necessary components
    missing_components = [arg for arg in model_args if arg not in loaded_data.keys() and arg not in optional_components]
    if missing_components:
        raise TypeError(f'Incomplete model parameters. Missing: {missing_components}')

    # create instance of the model class
    model = model_class(
        categories=loaded_data.get('categories', None),
        image_preprocessing=loaded_data['image_preprocessing'],
        image_augmentation=loaded_data['image_augmentation'],
    )
    model.tokenizer = loaded_data['tokenizer']
    model.text_vocab = loaded_data['text_vocab']
    model.category_vocab = loaded_data['category_vocab']
    model.classifier = loaded_data['classifier']
    model.eval_transforms = loaded_data['eval_transforms']
    model.train_transforms = loaded_data['train_transforms']
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
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if pt_path.endswith('lz4'):
        with open(pt_path, 'rb') as f:
            compressed = f.read()
        decompressed_data = lz4.frame.decompress(compressed)
        file_data = torch.load(io.BytesIO(decompressed_data), map_location=torch.device(device))

    else:
        with open(pt_path, 'rb') as f:  # todo check if we need to open 'rb' at all
            file_data = torch.load(pt_path, map_location=torch.device(device))

    if isinstance(file_data, dict):
        if pt_path.endswith('lz4'):
            file_data = _load_categorization_model(io.BytesIO(decompressed_data))
        else:
            file_data = _load_categorization_model(pt_path)
    else:
        if pt_path.endswith('lz4'):
            file_data = torch.load(io.BytesIO(decompressed_data), map_location=torch.device(device))
        else:
            with open(pt_path, 'rb') as f:
                file_data = torch.load(f, map_location=torch.device(device))

    return file_data
