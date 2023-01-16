"""Split a multi-Document file into a list of shorter documents."""
import abc
import cv2
import logging
import os
import sys
import torch

import numpy as np
import tensorflow as tf

from copy import deepcopy
from keras.applications.vgg19 import preprocess_input
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input, concatenate
from keras.models import Model
from pympler import asizeof
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from transformers import BertTokenizer, AutoModel, AutoConfig
from typing import List

from konfuzio_sdk.data import Document, Page, Category
from konfuzio_sdk.evaluate import FileSplittingEvaluation
from konfuzio_sdk.trainer.information_extraction import load_model, BaseModel
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.utils import get_timestamp, normalize_memory

logger = logging.getLogger(__name__)

tf.config.experimental_run_functions_eagerly(True)


class AbstractFileSplittingModel(BaseModel, metaclass=abc.ABCMeta):
    """Abstract class for the filesplitting model."""

    @abc.abstractmethod
    def __init__(self, categories: List[Category], *args, **kwargs):
        """
        Initialize the class.

        :param categories: A list of Categories to run training/prediction of the model on.
        :type categories: List[Category]
        """
        super().__init__()
        self.output_dir = None
        if not len(categories):
            raise ValueError("Cannot initialize ContextAwareFileSplittingModel on an empty list.")
        for category in categories:
            if not type(category) == Category:
                raise ValueError("All elements of the list have to be Categories.")
            if not len(category.documents()):
                raise ValueError(f'{category} does not have Documents and cannot be used for training.')
            if not len(category.test_documents()):
                raise ValueError(f'{category} does not have test Documents.')
        projects = set([category.project for category in categories])
        if len(projects) > 1:
            raise ValueError("All Categories have to belong to the same Project.")
        self.categories = categories
        self.project = self.categories[0].project  # we ensured that at least one Category is present
        self.documents = [document for category in self.categories for document in category.documents()]
        self.test_documents = [document for category in self.categories for document in category.test_documents()]

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the custom model on the training Documents."""  # there is no return

    @abc.abstractmethod
    def predict(self, page: Page) -> Page:
        """
        Take a Page as an input and reassign is_first_page attribute's value if necessary.

        :param page: A Page to label first or non-first.
        :type page: Page
        :return: Page.
        """

    @property
    def temp_pkl_file_path(self) -> str:
        """Generate a path for temporary pickle file."""
        temp_pkl_file_path = os.path.join(
            self.output_dir,
            f'{get_timestamp(konfuzio_format="%Y-%m-%d-%H-%M")}_{self.name_lower()}_{self.project.id_}_' f'tmp.pkl',
        )
        return temp_pkl_file_path

    @property
    def pkl_file_path(self) -> str:
        """Generate a path for a resulting pickle file."""
        pkl_file_path = os.path.join(
            self.output_dir,
            f'{get_timestamp(konfuzio_format="%Y-%m-%d-%H-%M")}_{self.name_lower()}_{self.project.id_}' f'.pkl',
        )
        return pkl_file_path

    def lose_weight(self):
        """Remove all data not necessary for prediction."""
        self.documents = None
        self.test_documents = None

    def reduce_model_weight(self):
        """Remove all non-strictly necessary parameters before saving."""
        self.lose_weight()
        self.tokenizer.lose_weight()

    def ensure_model_memory_usage_within_limit(self, max_ram):
        """
        Ensure that a model is not exceeding allowed max_ram.

        :param max_ram: Specify maximum memory usage condition to save model.
        """
        if not max_ram:
            max_ram = self.documents[0].project.max_ram

        max_ram = normalize_memory(max_ram)

        if max_ram and asizeof.asizeof(self) > max_ram:
            raise MemoryError(f"AI model memory use ({asizeof.asizeof(self)}) exceeds maximum ({max_ram=}).")

        sys.setrecursionlimit(99999999)

    def restore_category_documents_for_eval(self):
        """Restore Documents deleted when reducing weight in case there's evaluation needed."""
        self.documents = [document for category in self.categories for document in category.documents()]
        self.test_documents = [document for category in self.categories for document in category.test_documents()]


class FusionModel(AbstractFileSplittingModel):
    """
    Split a multi-Document file into a list of shorter documents based on model's prediction.

    We use an approach suggested by Guha et al.(2022) that incorporates steps for accepting separate visual and textual
    inputs and processing them independently via the VGG16 architecture and LegalBERT model which is essentially
    a BERT-type architecture trained on domain-specific data, and passing the resulting outputs together to
    a Multi-Layered Perceptron.

    Guha, A., Alahmadi, A., Samanta, D., Khan, M. Z., & Alahmadi, A. H. (2022).
    A Multi-Modal Approach to Digital Document Stream Segmentation for Title Insurance Domain.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9684474
    """

    def __init__(self, categories: List[Category], *args, **kwargs):
        """Initialize the Fusion filesplitting model."""
        logging.info('Initializing FusionModel.')
        super().__init__(categories=categories)
        self.name = self.__class__.__name__
        self.output_dir = self.project.model_folder
        self.train_txt_data = []
        self.train_img_data = None
        self.test_txt_data = []
        self.test_img_data = None
        self.train_labels = None
        self.test_labels = None
        self.input_shape = None
        self.model = None
        logger.info('Initializing BERT components of the FusionModel.')
        configuration = AutoConfig.from_pretrained('nlpaueb/legal-bert-base-uncased')
        configuration.num_labels = 2
        configuration.output_hidden_states = True
        self.bert_model = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased', config=configuration)
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'nlpaueb/legal-bert-base-uncased', do_lower_case=True, max_length=10000, padding="max_length", truncate=True
        )

    def _preprocess_documents(self, data: List[Document]) -> (List[str], List[str], List[int]):
        page_image_paths = []
        texts = []
        labels = []
        for doc in data:
            for page in doc.pages():
                page_image_paths.append(page.image_path)
                texts.append(page.text)
                if page.is_first_page:
                    labels.append(1)
                else:
                    labels.append(0)
        return page_image_paths, texts, labels

    def _otsu_binarization(self, pages: List[str]) -> List:
        images = []
        for img in pages:
            image = cv2.imread(img)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            images.append(image)
        return images

    def fit(self, *args, **kwargs):
        """Process the train and test data, initialize and fit the model."""
        logger.info('Fitting FusionModel.')
        for doc in self.documents + self.test_documents:
            doc.get_images()
        train_image_paths, train_texts, train_labels = self._preprocess_documents(self.documents)
        test_image_paths, test_texts, test_labels = self._preprocess_documents(self.test_documents)
        logger.info('Document preprocessing finished.')
        train_images = self._otsu_binarization(train_image_paths)
        test_images = self._otsu_binarization(test_image_paths)
        self.train_labels = tf.cast(np.asarray(train_labels).reshape((-1, 1)), tf.float32)
        self.test_labels = tf.cast(np.asarray(test_labels).reshape((-1, 1)), tf.float32)
        image_data_generator = ImageDataGenerator()
        train_data_generator = image_data_generator.flow(x=np.squeeze(train_images, axis=1), y=train_labels)
        self.train_img_data = np.concatenate(
            [train_data_generator.next()[0] for i in range(train_data_generator.__len__())]
        )
        test_data_generator = image_data_generator.flow(x=np.squeeze(test_images, axis=1), y=test_labels)
        self.test_img_data = np.concatenate(
            [test_data_generator.next()[0] for i in range(test_data_generator.__len__())]
        )
        logger.info('Image data preprocessing finished')
        for text in train_texts:
            inputs = self.bert_tokenizer(text, truncation=True, return_tensors='pt')
            with torch.no_grad():
                output = self.bert_model(**inputs)
            self.train_txt_data.append(output.pooler_output)
        self.train_txt_data = [np.asarray(x).astype('float32') for x in self.train_txt_data]
        self.train_txt_data = np.asarray(self.train_txt_data)
        for text in test_texts:
            inputs = self.bert_tokenizer(text, truncation=True, return_tensors='pt')
            with torch.no_grad():
                output = self.bert_model(**inputs)
            self.test_txt_data.append(output.pooler_output)
        self.input_shape = self.test_txt_data[0].shape
        self.test_txt_data = [np.asarray(x).astype('float32') for x in self.test_txt_data]
        self.test_txt_data = np.asarray(self.test_txt_data)
        logger.info('Text data preprocessing finished.')
        logger.info('FusionModel compiling started.')
        txt_input = Input(shape=self.input_shape, name='text')
        txt_x = Dense(units=768, activation="relu")(txt_input)
        txt_x = Flatten()(txt_x)
        txt_x = Dense(units=4096, activation="relu")(txt_x)
        img_input = Input(shape=(224, 224, 3), name='image')
        img_x = Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu")(
            img_input
        )
        img_x = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(img_x)
        img_x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(img_x)
        img_x = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(img_x)
        img_x = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(img_x)
        img_x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(img_x)
        img_x = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(img_x)
        img_x = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(img_x)
        img_x = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(img_x)
        img_x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(img_x)
        img_x = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(img_x)
        img_x = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(img_x)
        img_x = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(img_x)
        img_x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(img_x)
        img_x = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(img_x)
        img_x = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(img_x)
        img_x = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(img_x)
        img_x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(img_x)
        img_x = Flatten()(img_x)
        img_x = Dense(units=4096, activation="relu")(img_x)
        img_x = Dense(units=4096, activation="relu", name='img_outputs')(img_x)
        concatenated = concatenate([img_x, txt_x], axis=-1)
        x = Dense(50, input_shape=(8192,), activation='relu')(concatenated)
        x = Dense(50, activation='elu')(x)
        x = Dense(50, activation='elu')(x)
        output = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=[img_input, txt_input], outputs=output)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        logger.info('FusionModel compiling finished.')
        self.model.fit([self.train_img_data, self.train_txt_data], self.train_labels, epochs=10, verbose=1)
        logger.info('FusionModel fitting finished.')

    def predict(self, page: Page) -> Page:
        """
        Run prediction with the trained model.

        :param page: A Page to be predicted as first or non-first.
        :type page: Page
        :return: A Page with possible changes in is_first_page attribute value.
        """
        inputs = self.bert_tokenizer(page.text, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = self.bert_model(**inputs)
        txt_data = [output.pooler_output]
        txt_data = [np.asarray(x).astype('float32') for x in txt_data]
        txt_data = np.asarray(txt_data)
        image = cv2.imread(page.image_path)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        image_data_generator = ImageDataGenerator()
        data_generator = image_data_generator.flow(x=np.squeeze([image], axis=1))
        img_data = np.concatenate([data_generator.next()[0] for i in range(data_generator.__len__())])
        preprocessed = [img_data.reshape((1, 224, 224, 3)), txt_data.reshape((1, 1, 768))]
        pred = round(self.model.predict(preprocessed, verbose=0)[0, 0])
        if pred == 1:
            page.is_first_page = True
        else:
            page.is_first_page = False
        return page


class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    """Fallback definition of a File Splitting Model."""

    def __init__(self, categories: List[Category], tokenizer, *args, **kwargs):
        """
        Initialize the ContextAwareFileSplittingModel.

        :param categories: A list of Categories to run training/prediction of the model on.
        :type categories: List[Category]
        :param tokenizer: Tokenizer used for processing Documents on fitting when searching for exclusive first-page
        strings.
        :raises ValueError: When an empty list of Categories is passed into categories argument.
        :raises ValueError: When a list passed into categories contains elements other than Categories.
        :raises ValueError: When a list passed into categories contains at least one Category with no documents or test
        documents.
        :raises ValueError: When a list passed into categories contains Categories from different Projects.
        """
        super().__init__(categories=categories)
        self.name = self.__class__.__name__
        self.output_dir = self.project.model_folder
        self.tokenizer = tokenizer
        self.path = None

    def fit(self, allow_empty_categories: bool = False, *args, **kwargs):
        """
        Gather the strings exclusive for first Pages in a given stream of Documents.

        Exclusive means that each of these strings appear only on first Pages of Documents within a Category.

        :param allow_empty_categories: To allow returning empty list for a Category if no exclusive first-page strings
        were found during fitting (which means prediction would be impossible for a Category).
        :type allow_empty_categories: bool
        :raises ValueError: When allow_empty_categories is False and no exclusive first-page strings were found for
        at least one Category.
        """
        for category in self.categories:
            cur_first_page_strings = category.exclusive_first_page_strings(tokenizer=self.tokenizer)
            if not cur_first_page_strings:
                if allow_empty_categories:
                    logger.warning(
                        f'No exclusive first-page strings were found for {category}, so it will not be used '
                        f'at prediction.'
                    )
                else:
                    raise ValueError(f'No exclusive first-page strings were found for {category}.')

    def predict(self, page: Page) -> Page:
        """
        Take a Page as an input and predict it as first or non-first.

        :param page: A Page to receive first or non-first label.
        :type page: Page
        :raises ValueError: When at least one Category does not have exclusive_first_page_strings.
        :return: A Page with a newly predicted is_first_page attribute.
        """
        for category in self.categories:
            if not category.exclusive_first_page_strings:
                raise ValueError(f"Cannot run prediction as {category} does not have exclusive_first_page_strings.")
        page.is_first_page = False
        for category in self.categories:
            cur_first_page_strings = category.exclusive_first_page_strings(tokenizer=self.tokenizer)
            intersection = {span.offset_string for span in page.spans()}.intersection(cur_first_page_strings)
            if len(intersection) > 0:
                page.is_first_page = True
                break
        return page


class SplittingAI:
    """Split a given Document and return a list of resulting shorter Documents."""

    def __init__(self, model="", tokenizer=ConnectedTextTokenizer()):
        """
        Initialize the class.

        :param model: A path to an existing .cloudpickle model or to a previously trained instance of the model.
        :param tokenizer: A tokenizer to use with ContextAwareFileSplittingModel in case this instance was passed into
        model variable.
        """
        if isinstance(model, str):
            self.model = load_model(model)
        else:
            self.model = model
        if not issubclass(type(self.model), AbstractFileSplittingModel):
            raise ValueError("The model is not inheriting from AbstractFileSplittingModel class.")
        if isinstance(type(self.model), ContextAwareFileSplittingModel):
            self.use_context_aware_logic = True
            self.tokenizer = tokenizer
        else:
            self.use_context_aware_logic = False

    def _suggest_first_pages(self, document: Document, inplace: bool = False) -> List[Document]:
        """
        Run prediction on Document's Pages, predicting them to be first or non-first.

        :param document: A Document to run prediction on.
        :type document: Document
        :param inplace: To create a copy of a Document or to run prediction over the original one.
        :type inplace: bool
        :returns: A list with a single Document – for the sake of unity across different prediction methods' returns
        in the class.
        """
        if self.use_context_aware_logic:
            if inplace:
                document = self.tokenizer.tokenize(document)
            else:
                document = self.tokenizer.tokenize(deepcopy(document))
            for page in document.pages():
                self.model.predict(page)
        else:
            for page in document.pages():
                self.model.predict(page)
        return [document]

    def _suggest_page_split(self, document: Document) -> List[Document]:
        """
        Create a list of sub-Documents built from the original Document, split.

        :param document: A Document to run prediction on.
        :type document: Document
        :returns: A list of sub-Documents built from the original Document, based on model's prediction of first Pages.
        """
        suggested_splits = []
        if self.use_context_aware_logic:
            document = self.tokenizer.tokenize(deepcopy(document))
        for page in document.pages():
            if page.number == 1:
                suggested_splits.append(page)
            else:
                if self.model.predict(page).is_first_page:
                    suggested_splits.append(page)
        if len(suggested_splits) == 1:
            return [document]
        else:
            split_docs = []
            first_page = document.pages()[0]
            last_page = document.pages()[-1]
            for page_i, split_i in enumerate(suggested_splits):
                if page_i == 0:
                    split_docs.append(document.create_subdocument_from_page_range(first_page, split_i))
                elif page_i == len(split_docs):
                    split_docs.append(document.create_subdocument_from_page_range(split_i, last_page))
                else:
                    split_docs.append(
                        document.create_subdocument_from_page_range(suggested_splits[page_i - 1], split_i)
                    )
        return split_docs

    def propose_split_documents(
        self, document: Document, return_pages: bool = False, inplace: bool = False
    ) -> List[Document]:
        """
        Propose a set of resulting documents from a single Documents.

        :param document: An input Document to be split.
        :type document: Document
        :param inplace: Whether changes are applied to an initially passed Document, changing it, or to its deepcopy.
        :type inplace: bool
        :param return_pages: A flag to enable returning a copy of an old Document with Pages marked .is_first_page on
        splitting points instead of a set of sub-Documents.
        :type return_pages: bool
        :return: A list of suggested new sub-Documents built from the original Document or a list with a Document
        with Pages marked .is_first_page on splitting points.
        """
        if return_pages:
            processed = self._suggest_first_pages(document, inplace)
        else:
            processed = self._suggest_page_split(document)
        return processed

    def evaluate_full(self, use_training_docs: bool = False) -> FileSplittingEvaluation:
        """
        Evaluate the SplittingAI's performance.

        :param use_training_docs: If enabled, runs evaluation on the training data to define its quality; if disabled,
        runs evaluation on the test data.
        :type use_training_docs: bool
        :return: Evaluation information for the filesplitting context-aware logic.
        """
        pred_docs = []
        if not use_training_docs:
            original_docs = self.model.test_documents
        else:
            original_docs = self.model.documents
        for doc in original_docs:
            predictions = self.propose_split_documents(doc, return_pages=True)
            assert len(predictions) == 1
            pred_docs.append(predictions[0])
        self.full_evaluation = FileSplittingEvaluation(original_docs, pred_docs)
        return self.full_evaluation
