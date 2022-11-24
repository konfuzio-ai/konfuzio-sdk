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
import cv2
import logging
import torch
import pickle

import numpy as np
import tensorflow as tf

from keras.applications.vgg19 import preprocess_input
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input, concatenate
from keras.models import Model
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from transformers import BertTokenizer, AutoModel, AutoConfig
from typing import List

from konfuzio_sdk.data import Document, Page, Project

tf.config.experimental_run_functions_eagerly(True)


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

    def prepare_visual_textual_data(
        self, train_data: List[Document], test_data: List[Document], bert_model, bert_tokenizer
    ):
        """
        Prepare visual and textual inputs and transform them for feeding to the fusion model.

        :param train_data: Train dataset from the project.documents.
        :type train_data: list
        :param test_data: Test dataset from the project.test_documents.
        :type test_data: list
        :param bert_model: Initialized LegalBERT model.
        :param bert_tokenizer: Initialized BERTTokenizer.
        :return: Train and test visual inputs, train and test textual inputs, train and test labels, input shape for
        textual inputs.
        """
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
            inputs = bert_tokenizer(text, truncation=True, return_tensors='pt')
            with torch.no_grad():
                output = bert_model(**inputs)
            train_txt_data.append(output.pooler_output)
        train_txt_data = [np.asarray(x).astype('float32') for x in train_txt_data]
        train_txt_data = np.asarray(train_txt_data)
        test_txt_data = []
        for text in test_texts:
            inputs = bert_tokenizer(text, truncation=True, return_tensors='pt')
            with torch.no_grad():
                output = bert_model(**inputs)
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

    def init_bert(self):
        """Initialize BERT model and tokenizer."""
        configuration = AutoConfig.from_pretrained('nlpaueb/legal-bert-base-uncased')
        configuration.num_labels = 2
        configuration.output_hidden_states = True
        model = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased', config=configuration)
        tokenizer = BertTokenizer.from_pretrained(
            'nlpaueb/legal-bert-base-uncased', do_lower_case=True, max_length=10000, padding="max_length", truncate=True
        )
        return model, tokenizer

    def _predict_label(self, img_input, txt_input, model) -> int:
        pred = model.predict([img_input.reshape((1, 224, 224, 3)), txt_input.reshape((1, 1, 768))], verbose=0)
        return round(pred[0, 0])

    def calculate_metrics(self, model, img_inputs: List, txt_inputs: List, labels: List) -> (float, float, float):
        """
        Calculate precision, recall, and F1 measure for the trained model.

        :param model: The trained model.
        :param img_inputs: Processed visual inputs from the test dataset.
        :type img_inputs: list
        :param txt_inputs: Processed textual inputs from the test dataset.
        :type txt_inputs: list
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
            # model = load_model(model)
        else:
            bert_model, bert_tokenizer = self.init_bert()
            (
                train_img_data,
                train_txt_data,
                test_img_data,
                test_txt_data,
                train_labels,
                test_labels,
                input_shape,
            ) = self.prepare_visual_textual_data(self.train_data, self.test_data, bert_model, bert_tokenizer)
            model = self.init_model(input_shape)
            model.fit([train_img_data, train_txt_data], train_labels, epochs=10, verbose=1)
            pickler = open(self.project.model_folder + '/fusion.pickle', "wb")
            pickle.dump(model, pickler)
            pickler.close()
            loss, acc = model.evaluate([test_img_data, test_txt_data], test_labels, verbose=0)
            logging.info('Accuracy: {}'.format(acc * 100))
            precision, recall, f1 = self.calculate_metrics(model, test_img_data, test_txt_data, test_labels)
            logging.info('\n Precision: {} \n Recall: {} \n F1-score: {}'.format(precision, recall, f1))
        return model


class SplittingAI:
    """
    Split a given Document and return a list of resulting shorter Documents.

    For each page of the Document, a prediction is run to determine whether it is a first page or not. Based off on the
    predictions, the Document is split on pages predicted as first, and a set of new shorter Documents is produced.
    """

    def __init__(self, model_path: str, project_id=None):
        """
        Load fusion model, VGG16 model, BERT and BERTTokenizer.

        :param model_path: A path to the trained model, if exists.
        :type model_path: str
        :param project_id: Project from which the Documents eligible for splitting are taken.
        :type project_id: int
        """
        self.file_splitter = FileSplittingModel(project_id=project_id)
        self.project = Project(id_=project_id)
        if Path(model_path).exists():
            unpickler = open(self.project.model_folder + '/fusion.pickle', 'rb')
            self.model = pickle.load(unpickler)
            unpickler.close()
        else:
            logging.info('Model not found, starting training.')
            self.model = self.file_splitter.train()
        self.bert_model, self.tokenizer = self.file_splitter.init_bert()

    def _preprocess_inputs(self, text: str, image) -> List:
        inputs = self.tokenizer(text, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = self.bert_model(**inputs)
        txt_data = [output.pooler_output]
        txt_data = [np.asarray(x).astype('float32') for x in txt_data]
        txt_data = np.asarray(txt_data)
        image = cv2.imread(image)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        image_data_generator = ImageDataGenerator()
        data_generator = image_data_generator.flow(x=np.squeeze([image], axis=1))
        img_data = np.concatenate([data_generator.next()[0] for i in range(data_generator.__len__())])
        return [img_data.reshape((1, 224, 224, 3)), txt_data.reshape((1, 1, 768))]

    def _predict(self, text_input: str, img_input, model) -> int:
        preprocessed = self._preprocess_inputs(text_input, img_input)
        prediction = model.predict(preprocessed, verbose=0)
        return round(prediction[0, 0])

    def _create_doc_from_page_interval(self, original_doc: Document, start_page: Page, end_page: Page) -> Document:
        pages_text = original_doc.text[start_page.start_offset : end_page.end_offset]
        new_doc = Document(project=self.project, id_=None, text=pages_text)
        for page in original_doc.pages():
            if page.number in range(start_page.number, end_page.number):
                _ = Page(
                    document=new_doc,
                    start_offset=page.start_offset,
                    end_offset=page.end_offset,
                    page_number=page.number,
                )
        return new_doc

    def _suggest_page_split(self, document: Document) -> List[Document]:
        suggested_splits = []
        document.get_images()
        for page_i, page in enumerate(document.pages()):
            is_first_page = self._predict(page.text, page.image_path, self.model)
            if is_first_page:
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
                split_docs.append(self._create_doc_from_page_interval(document, split_docs[page_i - 1], split_i))
        return split_docs

    def propose_split_documents(self, document: Document) -> List[Document]:
        """
        Propose a set of resulting documents from a single Documents.

        :param document: An input Document to be split.
        :return: A list of suggested new sub-Documents built from the original Document.
        """
        split_docs = self._suggest_page_split(document)
        return split_docs
