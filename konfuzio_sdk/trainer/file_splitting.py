"""Split a multi-Document file into a list of shorter documents based on model's prediction."""
import cv2
import logging
import torch

import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.losses import categorical_crossentropy
from keras.models import Sequential, load_model
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from transformers import BertTokenizer, AutoModelForSequenceClassification, AutoConfig
from typing import List
from zipfile import ZipFile

from konfuzio_sdk.data import Document, Page, Project


class FusionModel:
    """Train a fusion model for correct splitting of files which contain multiple Documents."""

    def __init__(self, project_id: int, split_point: float = 0.5):
        """Initialize Project, training and testing data, and a split point for training dataset."""
        self.project = Project(id_=project_id)
        self.train_data = self.project.documents
        self.test_data = self.project.test_documents
        self.split_point = int(split_point * len(self.train_data))

    def _preprocess_documents(self, data: List[Document], path: str):
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
        Path("otsu_vgg16/{}/first_page".format(path)).mkdir(parents=True, exist_ok=True)
        Path("otsu_vgg16/{}/not_first_page".format(path)).mkdir(parents=True, exist_ok=True)
        return pages, texts, labels

    def _otsu_binarization(self, pages: List, labels: List[int], path):
        for img, label in zip(pages, labels):
            image = cv2.imread(img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image = cv2.resize(thresh1, (224, 224), interpolation=cv2.INTER_AREA)
            # cv2.imwrite(
            #     img.split('.')[0] + '_otsu.png', image
            # )
            if label == 0:
                cv2.imwrite(
                    'otsu_vgg16/{}/not_first_page/{}'.format(path, img.split('/')[-2] + '_' + img.split('/')[-1]), image
                )
            else:
                cv2.imwrite(
                    'otsu_vgg16/{}/first_page/{}'.format(path, img.split('/')[-2] + '_' + img.split('/')[-1]), image
                )

    def _prepare_visual_textual_data(
        self,
        train_data: List[Document],
        test_data: List[Document],
        split_point: int,
    ):
        for doc in train_data + test_data:
            doc.get_images()
        train_pages_1, train_texts_1, train_labels_1 = self._preprocess_documents(train_data[:split_point], 'train_1')
        train_pages_2, train_texts_2, train_labels_2 = self._preprocess_documents(train_data[split_point:], 'train_2')
        test_pages, test_texts, test_labels = self._preprocess_documents(test_data, 'test')
        self._otsu_binarization(train_pages_1, train_labels_1, 'train_1')
        self._otsu_binarization(train_pages_2, train_labels_2, 'train_2')
        self._otsu_binarization(test_pages, test_labels, 'test')
        return (
            train_texts_1,
            train_texts_2,
            test_texts,
            train_pages_1,
            train_pages_2,
            test_pages,
            train_labels_2,
            test_labels,
        )

    def _prepare_image_data_generator(self):
        image_data_generator = ImageDataGenerator()
        train_data_generator = image_data_generator.flow_from_directory(
            directory="otsu_vgg16/train_1", target_size=(224, 224)
        )
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
        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy'])
        return model

    def train_vgg(self, image_data_generator=None):
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

    def get_logits_vgg16(self, pages: List[str], model):
        """Transform input images into logits for MLP input."""
        logits = []
        for path in pages:
            img = load_img(path, target_size=(224, 224))
            img = np.asarray(img)
            img = np.expand_dims(img, axis=0)
            output = model.predict(img)
            logits.append(output)
        return logits

    def get_logits_bert(self, texts, tokenizer, model):
        """Transform input texts into logits for MLP input."""
        logits = []
        for text in texts:
            inputs = tokenizer(text, truncation=True, return_tensors='pt')
            with torch.no_grad():
                output = model(**inputs).logits
            pred = output.argmax().item()
            logits.append(pred)
        return logits

    def squash_logits(self, vgg_logits, bert_logits):
        """Squash image and text logits together for MLP input."""
        logits = []
        for logit_1, logit_2 in zip(vgg_logits, bert_logits):
            logits.append([logit_1[0][0], logit_1[0][1], logit_2, logit_2])
        return logits

    def _preprocess_mlp_inputs(self, train_logits, test_logits, train_labels, test_labels):
        Xtrain = np.array(train_logits)
        Xtest = np.array(test_logits)
        ytrain = np.array(train_labels)
        ytest = np.array(test_labels)
        input_shape = Xtest.shape[1]
        return Xtrain, Xtest, ytrain, ytest, input_shape

    def _predict_label(self, inputs, model):
        pred = model.predict(inputs, verbose=0)
        return round(pred[0, 0])

    def _calculate_metrics(self, model, inputs, labels):
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

    def train(self):
        """Preprocess data, train VGG16 and a MLP pipeline based on VGG16 and LegalBERT."""
        (
            train_texts_1,
            train_texts_2,
            test_texts,
            train_pages_1,
            train_pages_2,
            test_pages,
            train_labels_2,
            test_labels,
        ) = self._prepare_visual_textual_data(self.train_data, self.test_data, self.split_point)
        train_data_generator = self._prepare_image_data_generator()
        model_vgg = self.train_vgg(image_data_generator=train_data_generator)
        bert, tokenizer = self.init_bert()
        vgg16_train_logits = self.get_logits_vgg16(train_pages_2, model_vgg)
        vgg16_test_logits = self.get_logits_vgg16(test_pages, model_vgg)
        bert_train_logits = self.get_logits_bert(train_texts_2, tokenizer, bert)
        bert_test_logits = self.get_logits_bert(test_texts, tokenizer, bert)
        train_logits = self.squash_logits(vgg16_train_logits, bert_train_logits)
        test_logits = self.squash_logits(vgg16_test_logits, bert_test_logits)
        Xtrain, Xtest, ytrain, ytest, input_shape = self._preprocess_mlp_inputs(
            train_logits, test_logits, train_labels_2, test_labels
        )
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
        zip_obj = ZipFile(self.project.model_folder + '/splitting_ai_models.zip', 'w')
        zip_obj.write(self.project.model_folder + '/fusion.h5')
        zip_obj.write(self.project.model_folder + '/vgg16.h5')
        zip_obj.close()
        Path(self.project.model_folder + '/fusion.h5').unlink()
        Path(self.project.model_folder + '/vgg16.h5').unlink()
        return model


class SplittingAI:
    """Split a given Document and return a list of resulting shorter Documents."""

    def __init__(self, model_path: str, project_id=None):
        """Load fusion model, VGG16 model, BERT and BERTTokenizer."""
        self.file_splitter = FusionModel(project_id=project_id, split_point=0.5)
        self.project = Project(id_=project_id)
        if Path(model_path).exists():
            with ZipFile(self.project.model_folder + '/splitting_ai_models.zip', 'r') as zip_ref:
                zip_ref.extractall()
            #
            # input_zip = ZipFile(self.project.model_folder + '/splitting_ai_models.zip')
            # models = {name: input_zip.read(name) for name in input_zip.namelist()}
            self.model = load_model(self.project.model_folder + '/fusion.h5')
            self.vgg16 = load_model(self.project.model_folder + '/vgg16.h5')
            # with ZipFile(self.project.model_folder + '/splitting_ai_models.zip') as zip_file:
            #     self.model = load_model("fusion.h5")
            #     self.vgg16 = load_model('vgg16.h5')
            # self.model = load_model(model_path)
        else:
            logging.info('Model not found, starting training.')
            self.model = self.file_splitter.train()
        self.bert_model, self.tokenizer = self.file_splitter.init_bert()
        # if Path(vgg16_path).exists():
        #     self.vgg16 = load_model(vgg16_path)
        # else:
        #     train_data_generator = self.file_splitter._prepare_image_data_generator()
        #     self.vgg16 = self.file_splitter.train_vgg(train_data_generator)

    def _preprocess_inputs(self, text: str, image):
        text_logits = self.file_splitter.get_logits_bert([text], self.tokenizer, self.bert_model)
        img_logits = self.file_splitter.get_logits_vgg16([image], self.vgg16)
        logits = np.array(self.file_splitter.squash_logits(img_logits, text_logits))
        return logits

    def _predict(self, text_input, img_input, model):
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

    def propose_mappings(self, document: Document) -> List[Document]:
        """
        Propose a set of resulting documents from a single Documents.

        :param document: An input Document to be split.
        """
        split_docs = self._suggest_page_split(document)
        return split_docs
