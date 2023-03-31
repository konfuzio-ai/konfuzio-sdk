"""Process Documents that consist of several files and propose splitting them into the Sub-Documents accordingly."""
import abc
import logging
import os
import torch

import numpy as np
import tensorflow as tf

from copy import deepcopy
from inspect import signature
from PIL import Image
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from transformers import BertTokenizer, AutoModel, AutoConfig
from typing import List

from konfuzio_sdk.data import Document, Page, Category
from konfuzio_sdk.evaluate import FileSplittingEvaluation
from konfuzio_sdk.trainer.information_extraction import BaseModel
from konfuzio_sdk.utils import get_timestamp

logger = logging.getLogger(__name__)

# proper compiling of Multimodal File Splitting Model requires eager running instead of lazy
# because of multiple inputs (read more about eager vs lazy (graph) here)
# https://towardsdatascience.com/eager-execution-vs-graph-execution-which-is-better-38162ea4dbf6
tf.config.experimental_run_functions_eagerly(True)


class AbstractFileSplittingModel(BaseModel, metaclass=abc.ABCMeta):
    """Abstract class for the File Splitting model."""

    @abc.abstractmethod
    def __init__(self, categories: List[Category], *args, **kwargs):
        """
        Initialize the class.

        :param categories: A list of Categories to run training/prediction of the model on.
        :type categories: List[Category]
        """
        super().__init__()
        self.output_dir = None
        if not categories:
            raise ValueError("Cannot initialize ContextAwareFileSplittingModel class on an empty list.")
        for category in categories:
            if not isinstance(category, Category):
                raise ValueError("All elements of the list have to be Categories.")
        nonempty_categories = [category for category in categories if category.documents()]
        if not nonempty_categories:
            raise ValueError("At least one Category has to have Documents for training the model.")
        for category in nonempty_categories:
            if not category.test_documents():
                raise ValueError(f'{category} does not have test Documents.')
        self.categories = categories
        self.project = self.categories[0].project  # we ensured that at least one Category is present
        self.documents = [document for category in self.categories for document in category.documents()]
        self.test_documents = [document for category in self.categories for document in category.test_documents()]
        self.tokenizer = None
        self.requires_text = False
        self.requires_images = False

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
        """
        Generate a path for temporary pickle file.

        :returns: A string with the path.
        """
        temp_pkl_file_path = os.path.join(
            self.output_dir,
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
            self.output_dir,
            f'{get_timestamp()}_{self.project.id_}_{self.name_lower()}.pkl',
        )
        return pkl_file_path

    @staticmethod
    def has_compatible_interface(other) -> bool:
        """
        Validate that an instance of a File Splitting Model implements the same interface as AbstractFileSplittingModel.

        A File Splitting Model should implement methods with the same signature as:
        - AbstractFileSplittingModel.__init__
        - AbstractFileSplittingModel.predict
        - AbstractFileSplittingModel.fit
        - AbstractFileSplittingModel.check_is_ready

        :param other: An instance of a File Splitting Model to compare with.
        """
        try:
            return (
                signature(other.__init__).parameters['categories'].annotation._name == 'List'
                and signature(other.__init__).parameters['categories'].annotation.__args__[0].__name__ == 'Category'
                and signature(other.predict).parameters['page'].annotation.__name__ == 'Page'
                and signature(other.predict).return_annotation.__name__ == 'Page'
                and signature(other.fit)
                and signature(other.check_is_ready)
            )
        except KeyError:
            return False
        except AttributeError:
            return False


class MultimodalFileSplittingModel(AbstractFileSplittingModel):
    """
    Split a multi-Document file into a list of shorter Documents based on model's prediction.

    We use an approach suggested by Guha et al.(2022) that incorporates steps for accepting separate visual and textual
    inputs and processing them independently via the VGG19 architecture and LegalBERT model which is essentially
    a BERT-type architecture trained on domain-specific data, and passing the resulting outputs together to
    a Multi-Layered Perceptron.

    Guha, A., Alahmadi, A., Samanta, D., Khan, M. Z., & Alahmadi, A. H. (2022).
    A Multi-Modal Approach to Digital Document Stream Segmentation for Title Insurance Domain.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9684474
    """

    def __init__(
        self,
        categories: List[Category],
        text_processing_model: str = 'nlpaueb/legal-bert-base-uncased',
        scale: int = 2,
        *args,
        **kwargs,
    ):
        """
        Initialize the Multimodal File Splitting Model.

        :param categories: Categories from which Documents for training and testing are used.
        :type categories: List[Category]
        :param text_processing_model: A path to the HuggingFace model that is used for processing the textual
        data from the Documents, can be a path in the HuggingFace repo or a local path to a checkpoint of a pre-trained
        HuggingFace model. Default is LegalBERT.
        :type text_processing_model: str
        :param scale: A multiplier to define a number of units (neurons) in Dense layers of a model for image
        processing.
        :type scale: int
        """
        logging.info('Initializing Multimodal File Splitting Model.')
        super().__init__(categories=categories)
        self.output_dir = self.project.model_folder
        self.requires_images = True
        self.requires_text = True
        self.train_txt_data = []
        self.train_img_data = None
        self.test_txt_data = []
        self.test_img_data = None
        self.train_labels = None
        self.test_labels = None
        self.input_shape = None
        self.scale = scale
        self.model = None
        logger.info('Initializing BERT components of the Multimodal File Splitting Model.')
        configuration = AutoConfig.from_pretrained(text_processing_model)
        configuration.num_labels = 2
        configuration.output_hidden_states = True
        self.bert_model = AutoModel.from_pretrained(text_processing_model, config=configuration)
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            text_processing_model, do_lower_case=True, max_length=2000, padding="max_length", truncate=True
        )

    def reduce_model_weight(self):
        """Remove all non-strictly necessary parameters before saving."""
        self.project.lose_weight()

    def _preprocess_documents(self, data: List[Document]) -> (List[str], List[str], List[int]):
        """
        Take a list of Documents and extract paths to its Pages' images, texts and labels of first or non-first class.

        :param data: A list of Documents to preprocess.
        :type data: List[Document]
        :returns: Three lists of strings – paths to Pages' images, Pages' texts and Pages' labels.
        """
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

    def _image_transformation(self, page_image_paths: List[str]) -> List[np.ndarray]:
        """
        Take an image and transform it into the format acceptable by the model's architecture.

        :param page_image_paths: A list of Pages' images to be transformed.
        :type page_image_paths: List[str]
        :returns: A list of processed images.
        """
        images = []
        for page_image_path in page_image_paths:
            image = Image.open(page_image_path).convert('RGB')
            image = image.resize((224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = image[..., ::-1]  # replacement of keras's preprocess_input implementation because of dimensionality
            image[0] -= 103.939
            images.append(image)
        return images

    def fit(self, epochs: int = 10, use_gpu: bool = False, *args, **kwargs):
        """
        Process the train and test data, initialize and fit the model.

        :param epochs: A number of epochs to train a model on.
        :type epochs: int
        :param use_gpu: Run training on GPU if available.
        :type use_gpu: bool
        """
        logger.info('Fitting Multimodal File Splitting Model.')
        for doc in self.documents + self.test_documents:
            for page in doc.pages():
                if not os.path.exists(page.image_path):
                    page.get_image()
        train_image_paths, train_texts, train_labels = self._preprocess_documents(self.documents)
        test_image_paths, test_texts, test_labels = self._preprocess_documents(self.test_documents)
        logger.info('Document preprocessing finished.')
        train_images = self._image_transformation(train_image_paths)
        test_images = self._image_transformation(test_image_paths)
        # labels are transformed into numpy array, reshaped into arrays of len==1 and then into TF tensor of shape
        # (len(train_labels), 1) with elements of dtype==float32. this is needed to feed labels into the Multi-Layered
        # Perceptron as input.
        self.train_labels = tf.cast(np.asarray(train_labels).reshape((-1, 1)), tf.float32)
        self.test_labels = tf.cast(np.asarray(test_labels).reshape((-1, 1)), tf.float32)
        self.train_img_data = np.concatenate(train_images)
        self.test_img_data = np.concatenate(test_images)
        logger.info('Image data preprocessing finished.')
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
        logger.info('Multimodal File Splitting Model compiling started.')
        # we combine an output of a simplified VGG19 architecture for image processing (read more about it
        # at https://iq.opengenus.org/vgg19-architecture/) and an output of BERT in an MLP-like
        # architecture (read more about it at http://shorturl.at/puKN3). a scheme of our custom architecture can be
        # found at https://dev.konfuzio.com/sdk/tutorials.html#splitting-for-multi-file-documents-step-by-step-guide
        txt_input = Input(shape=self.input_shape, name='text')
        txt_x = Dense(units=768, activation="relu")(txt_input)
        txt_x = Flatten()(txt_x)
        txt_x = Dense(units=256 * self.scale, activation="relu")(txt_x)
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
        img_x = Flatten()(img_x)
        img_x = Dense(units=256 * self.scale, activation="relu")(img_x)
        img_x = Dense(units=256 * self.scale, activation="relu", name='img_outputs')(img_x)
        concatenated = Concatenate(axis=-1)([img_x, txt_x])
        x = Dense(50, input_shape=(512 * self.scale,), activation='relu')(concatenated)
        x = Dense(50, activation='elu')(x)
        x = Dense(50, activation='elu')(x)
        output = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=[img_input, txt_input], outputs=output)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        logger.info('Multimodal File Splitting Model compiling finished.')
        if not use_gpu:
            with tf.device('/cpu:0'):
                self.model.fit([self.train_img_data, self.train_txt_data], self.train_labels, epochs=epochs, verbose=1)
        else:
            if tf.config.list_physical_devices('GPU'):
                with tf.device('/gpu:0'):
                    self.model.fit(
                        [self.train_img_data, self.train_txt_data], self.train_labels, epochs=epochs, verbose=1
                    )
            else:
                raise ValueError('Fitting on the GPU is impossible because there is no GPU available on the device.')
        logger.info('Multimodal File Splitting Model fitting finished.')

    def predict(self, page: Page, use_gpu: bool = False) -> Page:
        """
        Run prediction with the trained model.

        :param page: A Page to be predicted as first or non-first.
        :type page: Page
        :param use_gpu: Run prediction on GPU if available.
        :type use_gpu: bool
        :return: A Page with possible changes in is_first_page attribute value.
        """
        self.check_is_ready()
        inputs = self.bert_tokenizer(page.text, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = self.bert_model(**inputs)
        txt_data = [output.pooler_output]
        txt_data = [np.asarray(x).astype('float32') for x in txt_data]
        txt_data = np.asarray(txt_data)
        image = Image.open(page.image_path).convert('RGB')
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = image[..., ::-1]
        image[0] -= 103.939
        img_data = np.concatenate([image])
        preprocessed = [img_data.reshape((1, 224, 224, 3)), txt_data.reshape((1, 1, 768))]
        if not use_gpu:
            with tf.device('/cpu:0'):
                prediction = self.model.predict(preprocessed, verbose=0)[0, 0]
        else:
            if tf.config.list_physical_devices('GPU'):
                with tf.device('/gpu:0'):
                    prediction = self.model.predict(preprocessed, verbose=0)[0, 0]
            else:
                raise ValueError('Predicting on the GPU is impossible because there is no GPU available on the device.')
        page.is_first_page_confidence = prediction
        if round(prediction) == 1:
            page.is_first_page = True
        else:
            page.is_first_page = False
        return page

    def check_is_ready(self):
        """
        Check if Multimodal File Splitting Model instance is ready for inference.

        A method checks that the instance of the Model has at least one Category passed as the input and that it
        is fitted to run prediction.

        :raises AttributeError: When no Categories are passed to the model.
        :raises AttributeError: When a model is not fitted to run a prediction.
        """
        if not self.categories:
            raise AttributeError(f'{self} requires Categories.')

        if not self.model:
            raise AttributeError(f'{self} has to be fitted before running a prediction.')


class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    """
    A File Splitting Model that uses a context-aware logic.

    Context-aware logic implies a rule-based approach that looks for common strings between the first Pages of all
    Category's Documents.
    """

    def __init__(self, categories: List[Category], tokenizer, *args, **kwargs):
        """
        Initialize the Context Aware File Splitting Model.

        :param categories: A list of Categories to run training/prediction of the model on.
        :type categories: List[Category]
        :param tokenizer: Tokenizer used for processing Documents on fitting when searching for exclusive first-page
        strings.
        :raises ValueError: When an empty list of Categories is passed into categories argument.
        :raises ValueError: When a list passed into categories contains elements other than Categories.
        :raises ValueError: When a list passed into categories contains at least one Category with no Documents or test
        Documents.
        """
        super().__init__(categories=categories)
        self.output_dir = self.project.model_folder
        self.tokenizer = tokenizer
        self.requires_text = True
        self.requires_images = False

    def fit(self, allow_empty_categories: bool = False, *args, **kwargs):
        """
        Gather the strings exclusive for first Pages in a given stream of Documents.

        Exclusive means that each of these strings appear only on first Pages of Documents within a Category.

        :param allow_empty_categories: To allow returning empty list for a Category if no exclusive first-page strings
        were found during fitting (which means prediction would be impossible for a Category).
        :type allow_empty_categories: bool
        :raises ValueError: When allow_empty_categories is False and no exclusive first-page strings were found for
        at least one Category.

        >>> from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
        >>> from konfuzio_sdk.data import Project
        >>> project = Project(id_=46)
        >>> tokenizer = ConnectedTextTokenizer()
        >>> model = ContextAwareFileSplittingModel(categories=project.categories, tokenizer=tokenizer).fit()
        """
        for category in self.categories:
            # method exclusive_first_page_strings fetches a set of first-page strings exclusive among the Documents
            # of a given Category. they can be found in _exclusive_first_page_strings attribute of a Category after
            # the method has been run. this is needed so that the information remains even if local variable
            # cur_first_page_strings is lost.
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
        Predict a Page as first or non-first.

        :param page: A Page to receive first or non-first label.
        :type page: Page
        :return: A Page with a newly predicted is_first_page attribute.

        >>> from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
        >>> from konfuzio_sdk.data import Project
        >>> project = Project(id_=46)
        >>> tokenizer = ConnectedTextTokenizer()
        >>> test_document = project.get_document_by_id(44865)
        >>> model = ContextAwareFileSplittingModel(categories=project.categories, tokenizer=tokenizer)
        >>> model.fit()
        >>> model.check_is_ready()
        >>> model.predict(model.tokenizer.tokenize(test_document).pages()[0]).is_first_page
        True
        """
        self.check_is_ready()
        page.is_first_page = False
        for category in self.categories:
            cur_first_page_strings = category.exclusive_first_page_strings(tokenizer=self.tokenizer)
            intersection = {span.offset_string.strip('\f').strip('\n') for span in page.spans()}.intersection(
                cur_first_page_strings
            )
            if len(intersection) > 0:
                page.is_first_page = True
                break
        page.is_first_page_confidence = 1
        return page

    def check_is_ready(self):
        """
        Check File Splitting Model is ready for inference.

        :raises AttributeError: When no Tokenizer or no Categories were passed.
        :raises ValueError: When no Categories have _exclusive_first_page_strings.
        """
        if self.tokenizer is None:
            raise AttributeError(f'{self} missing Tokenizer.')

        if not self.categories:
            raise AttributeError(f'{self} requires Categories.')

        empty_first_page_strings = [
            category
            for category in self.categories
            if not category.exclusive_first_page_strings(tokenizer=self.tokenizer)
        ]
        if len(empty_first_page_strings) == len(self.categories):
            raise ValueError(
                f"Cannot run prediction as none of the Categories in {self.project} have "
                f"_exclusive_first_page_strings."
            )


class SplittingAI:
    """Split a given Document and return a list of resulting shorter Documents."""

    def __init__(self, model):
        """
        Initialize the class.

        :param model: A previously trained instance of the File Splitting Model.
        :raises ValueError: When the model is not inheriting from AbstractFileSplittingModel class.
        """
        self.tokenizer = None
        if not AbstractFileSplittingModel.has_compatible_interface(model):
            raise ValueError("The model is not inheriting from AbstractFileSplittingModel class.")
        self.model = model
        if not self.model.requires_images:
            self.tokenizer = self.model.tokenizer

    def _suggest_first_pages(self, document: Document, inplace: bool = False) -> List[Document]:
        """
        Run prediction on Document's Pages, marking them as first or non-first.

        :param document: The Document to predict the Pages of.
        :type document: Document
        :param inplace: Whether to predict the Pages on the original Document or on a copy.
        :type inplace: bool
        :returns: A list containing the modified Document.
        """
        if not self.model.requires_images:
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
        Create a list of Sub-Documents built from the original Document, split.

        :param document: The Document to suggest Page splits for.
        :type document: Document
        :returns: A list of Sub-Documents created from the original Document, split at predicted first Pages.
        """
        suggested_splits = []

        if self.tokenizer:
            document_tokenized = self.tokenizer.tokenize(deepcopy(document))
        else:
            document_tokenized = document
        for page in document_tokenized.pages():
            if page.number == 1:
                suggested_splits.append(page)
            else:
                if self.model.predict(page).is_first_page:
                    suggested_splits.append(page)
        if len(suggested_splits) == 1:
            return [document]
        else:
            split_docs = []
            first_page = document_tokenized.pages()[0]
            last_page = document_tokenized.pages()[-1]
            for page_i, split_i in enumerate(suggested_splits):
                if page_i == 0:
                    split_docs.append(
                        document_tokenized.create_subdocument_from_page_range(
                            first_page, suggested_splits[page_i + 1], include=False
                        )
                    )
                elif page_i == len(suggested_splits) - 1:
                    split_docs.append(
                        document_tokenized.create_subdocument_from_page_range(split_i, last_page, include=True)
                    )
                else:
                    split_docs.append(
                        document_tokenized.create_subdocument_from_page_range(
                            split_i, suggested_splits[page_i + 1], include=False
                        )
                    )
        return split_docs

    def propose_split_documents(
        self, document: Document, return_pages: bool = False, inplace: bool = False
    ) -> List[Document]:
        """
        Propose a set of resulting Documents from a single Document.

        :param document: An input Document to be split.
        :type document: Document
        :param inplace: Whether changes are applied to the input Document, changing it, or to a deepcopy of it.
        :type inplace: bool
        :param return_pages: A flag to enable returning a copy of an old Document with Pages marked .is_first_page on
        splitting points instead of a set of Sub-Documents.
        :type return_pages: bool
        :return: A list of suggested new Sub-Documents built from the original Document or a list with a Document
        with Pages marked .is_first_page on splitting points.
        """
        if not document.category:
            raise AttributeError("A Document without Category cannot be split.")
        if self.model.requires_images:
            document.get_images()
        if return_pages:
            processed = self._suggest_first_pages(document, inplace)
        else:
            processed = self._suggest_page_split(document)
        return processed

    def evaluate_full(self, use_training_docs: bool = False) -> FileSplittingEvaluation:
        """
        Evaluate the Splitting AI's performance.

        :param use_training_docs: If enabled, runs evaluation on the training data to define its quality; if disabled,
        runs evaluation on the test data.
        :type use_training_docs: bool
        :return: Evaluation information for the model.
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
