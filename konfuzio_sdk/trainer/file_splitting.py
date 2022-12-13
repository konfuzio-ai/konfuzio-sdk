"""Split a multi-Document file into a list of shorter documents."""
import abc
import bz2
import cloudpickle
import cv2
import konfuzio_sdk
import logging
import os
import pathlib
import pickle
import shutil
import sys
import torch

import numpy as np
import tensorflow as tf

from copy import deepcopy
from keras.applications.vgg19 import preprocess_input
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input, concatenate
from keras.models import Model
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from transformers import BertTokenizer, AutoModel, AutoConfig
from typing import List, Union

from konfuzio_sdk.data import Document, Page, Project
from konfuzio_sdk.evaluate import FileSplittingEvaluation
from konfuzio_sdk.trainer.information_extraction import load_model
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.utils import get_timestamp

logger = logging.getLogger(__name__)

tf.config.experimental_run_functions_eagerly(True)


class AbstractFileSplittingModel(metaclass=abc.ABCMeta):
    """Abstract class for the filesplitting model."""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the class."""

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the custom model on the training Documents."""

    @abc.abstractmethod
    def save(self, model_path=""):
        """
        Save the trained model.

        :param model_path: Path to save the model to.
        :type model_path: str
        """

    @abc.abstractmethod
    def predict(self, page: Page) -> Page:
        """
        Take a Page as an input and predict if it is first or non-first.

        :param page: A Page to label first or non-first.
        :type page: Page
        :return: A Page with a predicted is_first_page == True or False.
        """


class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    """Fallback definition of a File Splitting Model."""

    def __init__(self, *args, **kwargs):
        """Initialize the ContextAwareFileSplittingModel."""
        self.train_data = None
        self.test_data = None
        self.categories = None
        self.tokenizer = None
        self.first_page_spans = None
        sys.setrecursionlimit(99999999)

    def fit(self, *args, **kwargs) -> dict:
        """
        Gather the Spans unique for first Pages in a given stream of Documents.

        :return: Dictionary with unique first-page Span sets by Category ID.
        """
        first_page_spans = {}
        for category in self.categories:
            cur_first_page_spans = []
            cur_non_first_page_spans = []
            for doc in category.documents():
                doc = deepcopy(doc)
                doc.category = category
                doc = self.tokenizer.tokenize(doc)
                for page in doc.pages():
                    if page.number == 1:
                        cur_first_page_spans.append({span.offset_string for span in page.spans()})
                    else:
                        cur_non_first_page_spans.append({span.offset_string for span in page.spans()})
            if not cur_first_page_spans:
                cur_first_page_spans.append(set())
            true_first_page_spans = set.intersection(*cur_first_page_spans)
            if not cur_non_first_page_spans:
                cur_non_first_page_spans.append(set())
            true_not_first_page_spans = set.intersection(*cur_non_first_page_spans)
            true_first_page_spans = true_first_page_spans - true_not_first_page_spans
            first_page_spans[category.id_] = true_first_page_spans
        self.first_page_spans = first_page_spans
        return first_page_spans

    def save(self, model_path="", include_konfuzio=True) -> str:
        """
        Save the resulting set of first-page Spans by Category.

        :param model_path: Path to save the set to.
        :type model_path: str
        :param include_konfuzio: Enables pickle serialization as a value, not as a reference (for more info, read
        https://github.com/cloudpipe/cloudpickle#overriding-pickles-serialization-mechanism-for-importable-constructs).
        :type include_konfuzio: bool
        """
        if include_konfuzio:
            cloudpickle.register_pickle_by_value(konfuzio_sdk)
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
        temp_pkl_file_path = os.path.join(model_path, f'{get_timestamp()}_first_page_spans_tmp.cloudpickle')
        pkl_file_path = os.path.join(model_path, f'{get_timestamp()}_first_page_spans.pkl')
        logger.info('Saving model with cloudpickle')
        with open(temp_pkl_file_path, 'wb') as f:
            cloudpickle.dump(self.first_page_spans, f)
        logger.info('Compressing model with bz2')
        with open(temp_pkl_file_path, 'rb') as input_f:
            with bz2.open(pkl_file_path, 'wb') as output_f:
                shutil.copyfileobj(input_f, output_f)
        logger.info('Deleting cloudpickle file')
        os.remove(temp_pkl_file_path)
        return pkl_file_path

    def predict(self, page: Page) -> Page:
        """
        Take a Page as an input and return 1 for a first Page and 0 for a non-first Page.

        :param page: A Page to receive first or non-first label.
        :type page: Page
        :return: A Page with or without is_first_page label.
        """
        for category in self.categories:
            intersection = {span.offset_string for span in page.spans()}.intersection(
                self.first_page_spans[category.id_]
            )
            if len(intersection) > 0:
                page.is_first_page = True
        return page


class FileSplittingModel:
    """Train a fusion model for correct splitting of files which contain multiple Documents.

    A model consists of two separate inputs for visual and textual data combined in a Multi-Layered
    Perceptron (MLP). Visual part is represented by VGG16 architecture and is trained on a first share of split training
    dataset. Textual part is represented by LegalBERT which is used without any training.
    Embeddings received from two of he models are squashed and the resulting vectors are fed as inputs to the MLP.

    The resulting trained model is saved in pickle, roughly 1.5 Gb in size.
    """

    def __init__(self, project_id: int):
        """
        Initialize Project, training and testing data.

        :param project_id: ID of the Project used for training the model.
        :type project_id: int
        """
        self.project = Project(id_=project_id)
        self.train_data = None
        self.test_data = None
        configuration = AutoConfig.from_pretrained('nlpaueb/legal-bert-base-uncased')
        configuration.num_labels = 2
        configuration.output_hidden_states = True
        self.bert_model = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased', config=configuration)
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'nlpaueb/legal-bert-base-uncased', do_lower_case=True, max_length=10000, padding="max_length", truncate=True
        )
        self.page_classifier = None

    def _preprocess_documents(self, data: List[Document]) -> (List[str], List[str], List[int]):
        pages = []
        texts = []
        labels = []
        for doc in data:
            for page in doc.pages():
                pages.append(page.image_path)
                texts.append(page.text)
                if page.number == 1:
                    labels.append(1)
                else:
                    labels.append(0)
        return pages, texts, labels

    def _otsu_binarization(self, pages: List[str]):
        images = []
        for img in pages:
            image = cv2.imread(img)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            images.append(image)
        return images

    def _prepare_visual_textual_data(self, train_data: List[Document], test_data: List[Document]):
        for doc in train_data + test_data:
            doc.get_images()
        train_pages, train_texts, train_labels = self._preprocess_documents(train_data)
        test_pages, test_texts, test_labels = self._preprocess_documents(test_data)
        train_images = self._otsu_binarization(train_pages)
        test_images = self._otsu_binarization(test_pages)
        train_labels = tf.cast(np.asarray(train_labels).reshape((-1, 1)), tf.float32)
        test_labels = tf.cast(np.asarray(test_labels).reshape((-1, 1)), tf.float32)
        image_data_generator = ImageDataGenerator()
        train_data_generator = image_data_generator.flow(x=np.squeeze(train_images, axis=1), y=train_labels)
        train_img_data = np.concatenate([train_data_generator.next()[0] for i in range(train_data_generator.__len__())])
        test_data_generator = image_data_generator.flow(x=np.squeeze(test_images, axis=1), y=test_labels)
        test_img_data = np.concatenate([test_data_generator.next()[0] for i in range(test_data_generator.__len__())])
        train_txt_data = []
        for text in train_texts:
            inputs = self.bert_tokenizer(text, truncation=True, return_tensors='pt')
            with torch.no_grad():
                output = self.bert_model(**inputs)
            train_txt_data.append(output.pooler_output)
        train_txt_data = [np.asarray(x).astype('float32') for x in train_txt_data]
        train_txt_data = np.asarray(train_txt_data)
        test_txt_data = []
        for text in test_texts:
            inputs = self.bert_tokenizer(text, truncation=True, return_tensors='pt')
            with torch.no_grad():
                output = self.bert_model(**inputs)
            test_txt_data.append(output.pooler_output)
        txt_input_shape = test_txt_data[0].shape
        test_txt_data = [np.asarray(x).astype('float32') for x in test_txt_data]
        test_txt_data = np.asarray(test_txt_data)
        return train_img_data, train_txt_data, test_img_data, test_txt_data, train_labels, test_labels, txt_input_shape

    def init_model(self, input_shape):
        """
        Initialize the fusion model.

        :param input_shape: Input shape for the textual part of the model.
        :type input_shape: tuple
        :return: A compiled fusion model.
        """
        txt_input = Input(shape=input_shape, name='text')
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
        model = Model(inputs=[img_input, txt_input], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def _predict_label(self, img_input, txt_input, model) -> int:
        pred = model.predict([img_input.reshape((1, 224, 224, 3)), txt_input.reshape((1, 1, 768))], verbose=0)
        return round(pred[0, 0])

    def calculate_metrics(self, model, img_inputs, txt_inputs, labels: List) -> (float, float, float):
        """
        Calculate precision, recall, and F1 measure for the trained model.

        :param model: The trained model.
        :param img_inputs: Processed visual inputs from the test dataset.
        :param txt_inputs: Processed textual inputs from the test dataset.
        :param labels: Labels from the test dataset.
        :type labels: list
        :return: Calculated precision, recall, and F1 measure.
        """
        true_positive = 0
        false_positive = 0
        false_negative = 0
        for img, txt, label in zip(img_inputs, txt_inputs, labels):
            pred = self._predict_label(img, txt, model)
            if label == 1 and pred == 1:
                true_positive += 1
            elif label == 1 and pred == 0:
                false_negative += 1
            elif label == 0 and pred == 1:
                false_positive += 1
        if true_positive + false_positive != 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0
        if true_positive + false_negative != 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        return precision, recall, f1

    def train(self):
        """
        Training or loading the trained model.

        :return: A trained fusion model.
        """
        if Path(self.project.model_folder + '/fusion.pickle').exists():
            unpickler = open(self.project.model_folder + '/fusion.pickle', 'rb')
            model = pickle.load(unpickler)
            unpickler.close()
        else:
            (
                train_img_data,
                train_txt_data,
                test_img_data,
                test_txt_data,
                train_labels,
                test_labels,
                input_shape,
            ) = self._prepare_visual_textual_data(self.train_data, self.test_data)
            model = self.init_model(input_shape)
            model.fit([train_img_data, train_txt_data], train_labels, epochs=10, verbose=1)
            self.page_classifier = model
            pickler = open(self.project.model_folder + '/fusion.pickle', "wb")
            pickle.dump(model, pickler)
            pickler.close()
            loss, acc = model.evaluate([test_img_data, test_txt_data], test_labels, verbose=0)
            logging.info('Accuracy: {}'.format(acc * 100))
            precision, recall, f1 = self.calculate_metrics(model, test_img_data, test_txt_data, test_labels)
            logging.info('\n Precision: {} \n Recall: {} \n F1-score: {}'.format(precision, recall, f1))
        return model


class SplittingAI:
    """Split a given Document and return a list of resulting shorter Documents."""

    def __init__(self, model=""):
        """
        Initialize the class.

        :param model: A path to an existing .cloudpickle model or to a previously trained instance of
        ContextAwareFileSplittingModel().
        """
        self.tokenizer = ConnectedTextTokenizer()
        if model is str:
            self.model = ContextAwareFileSplittingModel()
            self.model.first_page_spans = load_model(model)
        else:
            self.model = model

    def _create_doc_from_page_interval(self, original_doc: Document, start_page: Page, end_page: Page) -> Document:
        pages_text = original_doc.text[start_page.start_offset : end_page.end_offset]
        new_doc = Document(project=original_doc.project, id_=None, text=pages_text)
        for page in original_doc.pages():
            if page.number in range(start_page.number, end_page.number):
                _ = Page(
                    id_=None,
                    original_size=(page.height, page.width),
                    document=new_doc,
                    start_offset=page.start_offset,
                    end_offset=page.end_offset,
                    number=page.number,
                )
        return new_doc

    def _suggest_first_pages(self, document: Document) -> Document:
        new_doc = self.tokenizer.tokenize(deepcopy(document))
        for page in new_doc.pages():
            self.model.predict(page)
        return new_doc

    def _suggest_page_split(self, document: Document) -> List[Document]:
        suggested_splits = []
        document = self.tokenizer.tokenize(deepcopy(document))
        for page in document.pages():
            if page.number == 1:
                suggested_splits.append(page)
            else:
                if self.model.predict(page).is_first_page:
                    suggested_splits.append(page)
        split_docs = []
        first_page = document.pages()[0]
        last_page = document.pages()[-1]
        for page_i, split_i in enumerate(suggested_splits):
            if page_i == 0:
                split_docs.append(self._create_doc_from_page_interval(document, first_page, split_i))
            elif page_i == len(split_docs):
                split_docs.append(self._create_doc_from_page_interval(document, split_i, last_page))
            else:
                split_docs.append(self._create_doc_from_page_interval(document, suggested_splits[page_i - 1], split_i))
        return split_docs

    def propose_split_documents(self, document: Document, return_pages: bool = False) -> Union[Document, List]:
        """
        Propose a set of resulting documents from a single Documents.

        :param document: An input Document to be split.
        :type document: Document
        :param return_pages: A flag to enable returning a copy of an old Document with Pages marked .is_first_page on
        splitting points instead of a set of sub-Documents.
        :type return_pages: bool
        :return: A list of suggested new sub-Documents built from the original Document or a copy of an old Document
        with Pages marked .is_first_page on splitting points.
        """
        if return_pages:
            processed = self._suggest_first_pages(document)
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
        evaluation_list = []
        if not use_training_docs:
            evaluation_docs = self.model.test_data
        else:
            evaluation_docs = self.model.train_data
        for doc in evaluation_docs:
            doc.pages()[0].is_first_page = True
            pred = self.tokenizer.tokenize(deepcopy(doc))
            for page in pred.pages():
                if self.model.predict(page).is_first_page:
                    page.is_first_page = True
            evaluation_list.append((doc, pred))
        self.full_evaluation = FileSplittingEvaluation(evaluation_list)
        return self.full_evaluation
