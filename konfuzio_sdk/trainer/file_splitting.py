"""Split a multi-Document file into a list of shorter documents based on model's prediction."""
import cv2
import pickle
import torch

import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from nltk import word_tokenize
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from transformers import BertTokenizer, AutoModelForSequenceClassification, AutoConfig
from typing import List

from konfuzio_sdk.data import Document, Page, Project


class FusionModel:
    """Train a fusion model for correct splitting of multi-file documents."""

    def __init__(self, project_id: int, split_point: float = 0.5):
        """Initialize project, training and testing data, and a split point for training dataset."""
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
                if page.image_path.split('.')[-2] == 'page_1':
                    labels.append(1)
                else:
                    labels.append(0)
        Path("otsu_vgg16/{}/first_page".format(path)).mkdir(parents=True, exist_ok=True)
        Path("otsu_vgg16/{}/not_first_page".format(path)).mkdir(parents=True, exist_ok=True)
        return pages, texts, labels

    def _otsu_binarization(self, pages: List, labels: List[int], path: str):
        for img, label in zip(pages, labels):
            image = cv2.imread(img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image = cv2.resize(thresh1, (224, 224), interpolation=cv2.INTER_AREA)
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
        image_data_generator = ImageDataGenerator()
        train_data_generator = image_data_generator.flow_from_directory(
            directory="otsu_vgg16/train_1", target_size=(224, 224)
        )
        return (
            train_texts_1,
            train_texts_2,
            test_texts,
            train_pages_1,
            train_pages_2,
            test_pages,
            train_labels_2,
            test_labels,
            train_data_generator,
        )

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

    def _train_vgg(self, model, image_data_generator):
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
        return model

    def _init_bert(self):
        configuration = AutoConfig.from_pretrained('nlpaueb/legal-bert-base-uncased')
        configuration.num_labels = 2
        model = AutoModelForSequenceClassification.from_pretrained(
            'nlpaueb/legal-bert-base-uncased', config=configuration
        )
        tokenizer = BertTokenizer.from_pretrained(
            'nlpaueb/legal-bert-base-uncased', do_lower_case=True, max_length=10000, padding="max_length", truncate=True
        )
        return model, tokenizer

    def _get_logits_vgg16(self, pages: List[str], model):
        logits = []
        for path in pages:
            img = load_img(path, target_size=(224, 224))
            img = np.asarray(img)
            img = np.expand_dims(img, axis=0)
            output = model.predict(img)
            logits.append(output)
        return logits

    def _get_logits_bert(self, texts, tokenizer, model):
        logits = []
        for text in texts:
            inputs = tokenizer(text, truncation=True, return_tensors='pt')
            with torch.no_grad():
                output = model(**inputs).logits
            pred = output.argmax().item()
            logits.append(pred)
        return logits

    def _squash_logits(self, vgg_logits, bert_logits):
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
            train_data_generator,
        ) = self._prepare_visual_textual_data(self.train_data, self.test_data, self.split_point)
        model_vgg = self._init_vgg16()
        model_vgg = self._train_vgg(model_vgg, train_data_generator)
        bert, tokenizer = self._init_bert()
        vgg16_train_logits = self._get_logits_vgg16(train_pages_2, model_vgg)
        vgg16_test_logits = self._get_logits_vgg16(test_pages, model_vgg)
        bert_train_logits = self._get_logits_bert(train_texts_2, tokenizer, bert)
        bert_test_logits = self._get_logits_bert(test_texts, tokenizer, bert)
        train_logits = self._squash_logits(vgg16_train_logits, bert_train_logits)
        test_logits = self._squash_logits(vgg16_test_logits, bert_test_logits)
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
        return model


class PageSplitting:
    """Split a given document and return a list of resulting shorter documents."""

    def __init__(self, model_path: str):
        """Load model, tokenizer, vocabulary and categorization_pipeline."""
        self.load(model_path)

    def save(self) -> None:
        """Save model, tokenizer, and vocabulary used for document splitting."""
        pickle.dump((self.model, self.tokenizer, self.vocab))

    def load(self, path: str):
        """Load model, tokenizer, and vocabulary from a previously pickled file."""
        self.model, self.tokenizer, self.vocab = pickle.load(open(path))

    def _predict(self, page_text: str) -> bool:
        tokens = word_tokenize(page_text)
        tokens = [t for t in tokens if t in self.vocab]
        doc_text = ' '.join(tokens)
        encoded = self.tokenizer.texts_to_matrix([doc_text], mode='freq')
        predicted = self.model.predict(encoded, verbose=0)
        return bool(round(predicted[0, 0]))

    def _create_doc_from_page_interval(self, original_doc: Document, start_page: Page, end_page: Page) -> Document:
        pages_text = original_doc.text[start_page.start_offset : end_page.end_offset]
        new_doc = Document(id_=None, text=pages_text)
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
        for page_i, page in enumerate(document.pages()):
            is_first_page = self._predict(page.text)
            if is_first_page:
                suggested_splits.append(page_i)

        split_docs = []
        last_page = document.pages()[-1]
        for page_i, split_i in enumerate(suggested_splits):
            if page_i == 0:
                split_docs.append(self._create_doc_from_page_interval(document, page_i, split_i))
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
