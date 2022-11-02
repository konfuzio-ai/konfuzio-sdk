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
import tarfile
import torch

import numpy as np
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg19 import preprocess_input
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential, load_model
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from transformers import BertTokenizer, AutoModelForSequenceClassification, AutoConfig
from typing import List

from konfuzio_sdk.data import Document, Page, Project

tf.config.experimental_run_functions_eagerly(True)


class FileSplittingModel:
    """Train a fusion model for correct splitting of files which contain multiple Documents.

    A model consists of two separate inputs for visual and textual data combined together in a Multi-Layered
    Perceptron (MLP). Visual part is represented by VGG16 architecture and is trained on a first share of split training
    dataset. Textual part is represented by LegalBERT which is used without any training. Logits received from two of
    the models are squashed and the resulting logits are fed as inputs to the MLP.

    The resulting trained models (VGG16 and fusion model) are saved in .h5 and packaged into .tar.gz archive, roughly
    1.5 Gb in size.
    """

    def __init__(self, project_id: int, split_point: float = 0.5):
        """Initialize Project, training and testing data, and a split point for training dataset."""
        self.project = Project(id_=project_id)
        self.train_data = self.project.documents
        self.test_data = self.project.test_documents
        self.split_point = int(split_point * len(self.train_data))

    def _preprocess_documents(self, data: List[Document], path: str) -> (List[str], List[str], List[int]):
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
        # Path("otsu_vgg16/{}/first_page".format(path)).mkdir(parents=True, exist_ok=True)
        # Path("otsu_vgg16/{}/not_first_page".format(path)).mkdir(parents=True, exist_ok=True)
        return pages, texts, labels

    def _otsu_binarization(self, pages: List[str]) -> None:
        images = []
        for img in pages:
            image = cv2.imread(img)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            images.append(image)
        return images

    def _prepare_visual_textual_data(
        self,
        train_data: List[Document],
        test_data: List[Document],
        split_point: int,
    ) -> (List[str], List[str], List[str], List[str], List[str], List[str], List[int], List[int]):
        for doc in train_data + test_data:
            doc.get_images()
        train_pages_1, train_texts_1, train_labels_1 = self._preprocess_documents(train_data[:split_point], 'train_1')
        train_pages_2, train_texts_2, train_labels_2 = self._preprocess_documents(train_data[split_point:], 'train_2')
        test_pages, test_texts, test_labels = self._preprocess_documents(test_data, 'test')
        train_images_1 = self._otsu_binarization(train_pages_1)
        train_images_2 = self._otsu_binarization(train_pages_2)
        test_images = self._otsu_binarization(test_pages)
        return (
            train_texts_1,
            train_texts_2,
            test_texts,
            train_pages_1,
            train_pages_2,
            test_pages,
            train_labels_1,
            train_labels_2,
            test_labels,
            train_images_1,
            train_images_2,
            test_images,
        )

    def _prepare_image_data_generator(self, images, labels):
        labels = np.asarray(labels).astype('float32').reshape((-1, 1))
        image_data_generator = ImageDataGenerator()
        train_data_generator = image_data_generator.flow(x=np.squeeze(images, axis=1), y=labels)
        return train_data_generator

    def _init_vgg16(self):
        model = Sequential()
        model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=2, activation="softmax"))
        model.add(Flatten())
        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_vgg(self, image_data_generator: ImageDataGenerator = None):
        """Training or loading trained VGG16 model."""
        if Path(self.project.model_folder + '/vgg16.h5').exists():
            model = load_model(self.project.model_folder + '/vgg16.h5')
        else:
            model = self._init_vgg16()
            checkpoint = ModelCheckpoint(
                "vgg16.h5",
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                period=1,
            )
            early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
            model.fit_generator(
                steps_per_epoch=100,
                generator=image_data_generator,
                validation_steps=10,
                epochs=100,
                callbacks=[checkpoint, early],
            )
            model.save(self.project.model_folder + '/vgg16.h5')
        return model

    def init_bert(self):
        """Initialize BERT model and tokenizer."""
        configuration = AutoConfig.from_pretrained('nlpaueb/legal-bert-base-uncased')
        configuration.num_labels = 2
        model = AutoModelForSequenceClassification.from_pretrained(
            'nlpaueb/legal-bert-base-uncased', config=configuration
        )
        tokenizer = BertTokenizer.from_pretrained(
            'nlpaueb/legal-bert-base-uncased', do_lower_case=True, max_length=10000, padding="max_length", truncate=True
        )
        return model, tokenizer

    def get_logits_vgg16(self, images: List, model) -> List:
        """Transform input images into logits for MLP input."""
        logits = []
        for image in images:
            # img = load_img(path, target_size=(224, 224))
            # img = np.asarray(img)
            # img = np.expand_dims(img, axis=0)
            output = model.predict(image)
            logits.append(output)
        return logits

    def get_logits_bert(self, texts: List[str], tokenizer, model) -> List:
        """Transform input texts into logits for MLP input."""
        logits = []
        for text in texts:
            inputs = tokenizer(text, truncation=True, return_tensors='pt')
            with torch.no_grad():
                output = model(**inputs).logits
            pred = output.argmax().item()
            logits.append(pred)
        return logits

    def squash_logits(self, vgg_logits: List, bert_logits: List) -> List:
        """Squash image and text logits together for MLP input."""
        logits = []
        for logit_1, logit_2 in zip(vgg_logits, bert_logits):
            logits.append([logit_1[0][0], logit_1[0][1], logit_2, logit_2])
        return logits

    def _preprocess_mlp_inputs(
        self, train_logits: List, test_logits: List, train_labels: List[int], test_labels: List[int]
    ):
        Xtrain = np.array(train_logits)
        Xtest = np.array(test_logits)
        ytrain = np.array(train_labels)
        ytest = np.array(test_labels)
        input_shape = Xtest.shape[1]
        return Xtrain, Xtest, ytrain, ytest, input_shape

    def _predict_label(self, inputs, model) -> int:
        pred = model.predict(inputs, verbose=0)
        return round(pred[0, 0])

    def _calculate_metrics(self, model, inputs: List, labels: List) -> (float, float, float):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        for i, test in zip(labels, inputs):
            pred = self._predict_label(test.reshape((1, 4)), model)
            if i == 1 and pred == 1:
                true_positive += 1
            elif i == 1 and pred == 0:
                false_negative += 1
            elif i == 0 and pred == 1:
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

    def vgg16_preprocess_and_train(self, train_data: List[Document], test_data: List[Document], split_point: float):
        """Preprocess data and train VGG16 as a public method."""
        (
            train_texts_1,
            train_texts_2,
            test_texts,
            train_pages_1,
            train_pages_2,
            test_pages,
            train_labels_1,
            train_labels_2,
            test_labels,
            train_images_1,
            train_images_2,
            test_images,
        ) = self._prepare_visual_textual_data(train_data, test_data, split_point)
        train_data_generator = self._prepare_image_data_generator(train_images_1, train_labels_1)
        model_vgg = self.train_vgg(image_data_generator=train_data_generator)
        return model_vgg

    def prepare_mlp_inputs(
        self,
        train_data: List[Document],
        test_data: List[Document],
        split_point: float,
        model_vgg16,
        bert_model,
        bert_tokenizer,
    ):
        """Prepare data for feeding into an MLP as inputs."""
        (
            train_texts_1,
            train_texts_2,
            test_texts,
            train_pages_1,
            train_pages_2,
            test_pages,
            train_labels_1,
            train_labels_2,
            test_labels,
            train_images_1,
            train_images_2,
            test_images,
        ) = self._prepare_visual_textual_data(train_data, test_data, split_point)
        vgg16_train_logits = self.get_logits_vgg16(train_images_2, model_vgg16)
        vgg16_test_logits = self.get_logits_vgg16(test_images, model_vgg16)
        bert_train_logits = self.get_logits_bert(train_texts_2, bert_tokenizer, bert_model)
        bert_test_logits = self.get_logits_bert(test_texts, bert_tokenizer, bert_model)
        train_logits = self.squash_logits(vgg16_train_logits, bert_train_logits)
        test_logits = self.squash_logits(vgg16_test_logits, bert_test_logits)
        Xtrain, Xtest, ytrain, ytest, input_shape = self._preprocess_mlp_inputs(
            train_logits, test_logits, train_labels_2, test_labels
        )
        return Xtrain, Xtest, ytrain, ytest, input_shape

    def run_mlp(self, Xtrain, Xtest, ytrain, ytest, input_shape):
        """Compile, run, evaluate, and save an MLP architecture."""
        model = Sequential()
        model.add(Dense(50, input_shape=(input_shape,), activation='relu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(Xtrain, ytrain, epochs=100, verbose=2)
        model.save(self.project.model_folder + '/fusion.h5')
        loss, acc = model.evaluate(Xtest, ytest, verbose=0)
        logging.info('Accuracy: {}'.format(acc * 100))
        precision, recall, f1 = self._calculate_metrics(model, Xtest, ytest)
        logging.info('\n Precision: {} \n Recall: {} \n F1-score: {}'.format(precision, recall, f1))
        tar = tarfile.open(self.project.model_folder + "/splitting_ai_models.tar.gz", "w:gz")
        tar.add(self.project.model_folder + '/fusion.h5')
        tar.add(self.project.model_folder + '/vgg16.h5')
        tar.close()
        return model

    def train(self):
        """Preprocess data, train VGG16 and an MLP pipeline based on VGG16 and LegalBERT."""
        model_vgg16 = self.vgg16_preprocess_and_train(self.train_data, self.test_data, self.split_point)
        bert, tokenizer = self.init_bert()
        Xtrain, Xtest, ytrain, ytest, input_shape = self.prepare_mlp_inputs(
            self.train_data, self.test_data, self.split_point, model_vgg16, bert, tokenizer
        )
        model = self.run_mlp(Xtrain, Xtest, ytrain, ytest, input_shape)
        Path(self.project.model_folder + '/fusion.h5').unlink()
        Path(self.project.model_folder + '/vgg16.h5').unlink()
        return model


class SplittingAI:
    """
    Split a given Document and return a list of resulting shorter Documents.

    For each page of the Document, a prediction is run to determine whether it is a first page or not. Based off on the
    predictions, the Document is split on pages predicted as first, and a set of new shorter Documents is produced.
    """

    def __init__(self, model_path: str, project_id=None):
        """Load fusion model, VGG16 model, BERT and BERTTokenizer."""
        self.file_splitter = FileSplittingModel(project_id=project_id, split_point=0.5)
        self.project = Project(id_=project_id)
        if Path(model_path).exists():
            tar = tarfile.open(self.project.model_folder + '/splitting_ai_models.tar.gz', "r:gz")
            tar.extractall()
            self.model = load_model(self.project.model_folder + '/fusion.h5')
            self.vgg16 = load_model(self.project.model_folder + '/vgg16.h5')
        else:
            logging.info('Model not found, starting training.')
            self.model = self.file_splitter.train()
        self.bert_model, self.tokenizer = self.file_splitter.init_bert()

    def _preprocess_inputs(self, text: str, image) -> List:
        text_logits = self.file_splitter.get_logits_bert([text], self.tokenizer, self.bert_model)
        image = cv2.imread(image)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        img_logits = self.file_splitter.get_logits_vgg16([image], self.vgg16)
        logits = np.array(self.file_splitter.squash_logits(img_logits, text_logits))
        return logits

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
        """
        split_docs = self._suggest_page_split(document)
        return split_docs
