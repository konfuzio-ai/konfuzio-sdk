"""Process Documents that consist of several files and propose splitting them into the Sub-Documents accordingly."""
import abc
import logging
import os
import PIL
import time

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
)
from torch import nn
from datasets import Dataset
import evaluate

import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from copy import deepcopy
from inspect import signature
from typing import List, Union

from konfuzio_sdk.data import Document, Page, Category
from konfuzio_sdk.extras import torch, tensorflow as tf
from konfuzio_sdk.evaluate import FileSplittingEvaluation
from konfuzio_sdk.trainer.information_extraction import BaseModel
from konfuzio_sdk.utils import get_timestamp

logger = logging.getLogger(__name__)


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
                raise ValueError(f"{category} does not have test Documents.")
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
            f"{get_timestamp()}_{self.project.id_}_{self.name_lower()}_tmp.pkl",
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
            f"{get_timestamp()}_{self.project.id_}_{self.name_lower()}.pkl",
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
                signature(other.__init__).parameters["categories"].annotation._name == "List"
                and signature(other.__init__).parameters["categories"].annotation.__args__[0].__name__ == "Category"
                and signature(other.predict).parameters["page"].annotation.__name__ == "Page"
                and signature(other.predict).return_annotation.__name__ == "Page"
                and signature(other.fit)
                and signature(other.check_is_ready)
            )
        except KeyError:
            return False
        except AttributeError:
            return False

    @staticmethod
    def load_model(pickle_path: str, max_ram: Union[None, str] = None):
        """
        Load the model and check if it has the interface compatible with the class.

        :param pickle_path: Path to the pickled model.
        :type pickle_path: str
        :raises FileNotFoundError: If the path is invalid.
        :raises OSError: When the data is corrupted or invalid and cannot be loaded.
        :raises TypeError: When the loaded pickle isn't recognized as a Konfuzio AI model.
        :return: File Splitting AI model.
        """
        model = super(AbstractFileSplittingModel, AbstractFileSplittingModel).load_model(pickle_path, max_ram)
        if not AbstractFileSplittingModel.has_compatible_interface(model):
            raise TypeError(
                "Loaded model's interface is not compatible with any AIs. Please provide a model that has all the "
                "abstract methods implemented."
            )
        return model


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
        text_processing_model: str = "nlpaueb/legal-bert-small-uncased",
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
        logging.info("Initializing Multimodal File Splitting Model.")
        super().__init__(categories=categories)
        # proper compiling of Multimodal File Splitting Model requires eager running instead of lazy
        # because of multiple inputs (read more about eager vs lazy (graph) here)
        # https://towardsdatascience.com/eager-execution-vs-graph-execution-which-is-better-38162ea4dbf6
        tf.config.experimental_run_functions_eagerly(True)
        self.output_dir = self.project.model_folder
        self.requires_images = False
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
        logger.info("Initializing BERT components of the Multimodal File Splitting Model.")
        self.model_name = "distilbert-base-uncased"
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def reduce_model_weight(self):
        """Remove all non-strictly necessary parameters before saving."""
        self.project.lose_weight()

    def _preprocess_documents(
        self, data: List[Document], return_images: bool = False
    ) -> (List[PIL.Image.Image], List[str], List[int]):
        """
        Take a list of Documents and obtain Pages' images, texts and labels of first or non-first class.

        :param data: A list of Documents to preprocess.
        :type data: List[Document]
        :returns: Three lists – Pages' images, Pages' texts and Pages' labels.
        """
        if return_images:
            page_images = []
        texts = []
        labels = []
        for doc in data:
            for page in doc.pages():
                if return_images:
                    page_images.append(page.get_image())
                texts.append(page.text)
                if page.is_first_page:
                    labels.append(1)
                else:
                    labels.append(0)
        if return_images:
            return page_images, texts, labels
        return texts, labels

    def fit(
        self,
        epochs: int = 1,
        use_gpu: bool = False,
        eval_batch_size: int = 128,
        train_batch_size: int = 16,
        *args,
        **kwargs,
    ):
        """Process the train and test data, initialize and fit the model."""
        logger.info("Fitting Textual File Splitting Model.")
        logger.info("training documents:")
        print([doc.id_ for doc in self.documents])
        logger.info("testing documents:")
        print([doc.id_ for doc in self.test_documents])
        print("=" * 50)
        logger.info(f"Length of training documents: {len(self.documents)}")
        logger.info(f"Length of testing documents: {len(self.test_documents)}")
        logger.info("Preprocessing training & test documents")
        train_texts, train_labels = self._preprocess_documents(self.documents, return_images=False)
        test_texts, test_labels = self._preprocess_documents(self.test_documents, return_images=False)
        logger.info("Document preprocessing finished.")
        print("=" * 50)
        logger.info("Creating datasets")
        train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
        test_df = pd.DataFrame({"text": test_texts, "label": test_labels})
        # Convert to Dataset objects
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        # Calculate class weights to solve unbalanced dataset problem
        class_weights = compute_class_weight("balanced", classes=[0, 1], y=train_labels)
        # defining tokenizer
        tokenizer = self.bert_tokenizer
        # defining metric
        metric = evaluate.load("f1")

        # utility functions
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels, average="macro")

        print("=" * 50)
        logger.info("Tokenizing datasets")
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        print("=" * 50)
        logger.info("Loading model")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        training_args = TrainingArguments(
            output_dir="splitting_ai_trainer",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            learning_rate=1e-4,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=epochs,
            weight_decay=0,
        )
        print("=" * 50)
        logger.info(f"[{time.ctime(time.time())}]\tStarting Training...")
        logger.info("\nConfiguration to be used for Training:")
        logger.info(f"\nclass weights for the training dataset: {[f'{weight:.2e}' for weight in class_weights]}")
        logger.info(f"Number of epochs: {epochs}")
        logger.info(f"Batch size for training: {train_batch_size}")
        logger.info(f"Batch size for evaluation: {eval_batch_size}")
        logger.info(f"Learning rate: {training_args.learning_rate:.2e}\n")

        # custom trainer with custom loss to leverage class weights

        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                # forward pass
                outputs = model(**inputs)
                logits = outputs.get("logits")
                # compute custom loss (suppose one has 3 labels with different weights)
                loss_fct = nn.CrossEntropyLoss(
                    weight=torch.tensor(class_weights, device=model.device, dtype=torch.float)
                )
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        logger.info(f"[{time.ctime(time.time())}]\t🎉 Textual File Splitting Model fitting finished.")
        print("=" * 50)
        logger.info(f"[{time.ctime(time.time())}]\tComputing AI Quality.")
        evaluation_results = trainer.evaluate()
        self.model = trainer.model
        logger.info(f"[{time.ctime(time.time())}]\tTextual File Splitting Model Evaluation finished.")
        print("=" * 50)
        return evaluation_results

    def predict(self, page: Page, use_gpu: bool = False, previous_page: Page = None) -> Page:
        """
        Run prediction with the trained model.

        :param page: A Page to be predicted as first or non-first.
        :type page: Page
        :param use_gpu: Run prediction on GPU if available.
        :type use_gpu: bool
        :return: A Page with possible changes in is_first_page attribute value.
        """
        self.check_is_ready()
        tokenized_text = self.bert_tokenizer(page.text, truncation=True, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            self.model.eval()
            output = self.model(**tokenized_text)
        logits = output.logits

        # apply a softmax to get probabilities
        probabilities = torch.softmax(logits, dim=1)

        # Extract the probability of the 'is_first_page' being True
        predicted_prob_is_first = probabilities[:, 1].item()

        # if the previous page is not None and is empty, increase the probability of the current page being the first
        if previous_page is not None and len(previous_page.text.strip()) == 0:
            predicted_prob_is_first += 0.4

        # Determine the predicted label based on a threshold (e.g., 0.5)
        predicted_is_first = predicted_prob_is_first >= 0.5

        # Update the 'is_first_page' & 'is_first_page_confidence' attributes of the Page object
        page.is_first_page = predicted_is_first
        page.is_first_page_confidence = predicted_prob_is_first
        return page

    def remove_dependencies(self):
        """
        Remove dependencies before saving.

        This is needed for proper saving of the model in lz4 compressed format – if the dependencies are not removed,
        the resulting pickle will be impossible to load.
        """
        globals()["torch"] = None
        globals()["tf"] = None
        globals()["transformers"] = None

        del globals()["torch"]
        del globals()["tf"]
        del globals()["transformers"]

    def restore_dependencies(self):
        """
        Restore removed dependencies after loading.

        This is needed for proper functioning of a loaded model because we have previously removed these dependencies
        upon saving the model.
        """
        from konfuzio_sdk.extras import torch, tensorflow as tf, transformers

        globals()["torch"] = torch
        globals()["tf"] = tf
        globals()["transformers"] = transformers

    def check_is_ready(self):
        """
        Check if Multimodal File Splitting Model instance is ready for inference.

        A method checks that the instance of the Model has at least one Category passed as the input and that it
        is fitted to run prediction.

        :raises AttributeError: When no Categories are passed to the model.
        :raises AttributeError: When a model is not fitted to run a prediction.
        """
        if not self.categories:
            raise AttributeError(f"{self} requires Categories.")

        if not self.model:
            raise AttributeError(f"{self} has to be fitted before running a prediction.")

        self.restore_dependencies()


# begin class (this and further comments are for the documentation)
class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    """
    A File Splitting Model that uses a context-aware logic.

    Context-aware logic implies a rule-based approach that looks for common strings between the first Pages of all
    Category's Documents.
    """

    # begin init
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

    # end init

    # begin fit
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
                        f"No exclusive first-page strings were found for {category}, so it will not be used "
                        f"at prediction."
                    )
                else:
                    raise ValueError(f"No exclusive first-page strings were found for {category}.")

    # end fit

    # begin predict
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
            intersection = {span.offset_string.strip("\f").strip("\n") for span in page.spans()}.intersection(
                cur_first_page_strings
            )
            if len(intersection) > 0:
                page.is_first_page = True
                break
        page.is_first_page_confidence = 1
        return page

    # end predict

    # begin check
    def check_is_ready(self):
        """
        Check File Splitting Model is ready for inference.

        :raises AttributeError: When no Tokenizer or no Categories were passed.
        :raises ValueError: When no Categories have _exclusive_first_page_strings.
        """
        if self.tokenizer is None:
            raise AttributeError(f"{self} missing Tokenizer.")

        if not self.categories:
            raise AttributeError(f"{self} requires Categories.")

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

    # end check


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
        if inplace:
            processed_document = document
        else:
            processed_document = deepcopy(document)

        if not self.model.requires_images:
            # TODO: delete it, we don't need tokenizer
            # no, we need tokenizer for ContextAwareFileSplittingModel.
            if self.model.name == "ContextAwareFileSplittingModel":
                processed_document = self.tokenizer.tokenize(processed_document)
            # we set a Page's Category explicitly because we don't want to lose original Page's Category information
            # because by default a Page is assigned a Category of a Document, and they are not necessarily the same
            for index, page in enumerate(processed_document.pages()):
                previous_page = None if index == 0 else processed_document.pages()[index - 1]
                if self.model.name != "ContextAwareFileSplittingModel":
                    self.model.predict(page=page, previous_page=previous_page)
                else:
                    self.model.predict(page=page)
                # Why is done the in both cases?
                page.set_category(page.get_original_page().category)
        else:
            for index, page in enumerate(processed_document.pages()):
                previous_page = None if index == 0 else processed_document.pages()[index - 1]
                self.model.predict(page=page, previous_page=previous_page)
                # Why is done the in both cases?
                page.set_category(page.get_original_page().category)
        return [processed_document]

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
        for index, page in enumerate(document_tokenized.pages()):
            previous_page = None if index == 0 else document_tokenized.pages()[index - 1]
            if self.model.name != "ContextAwareFileSplittingModel":
                prediction = self.model.predict(page=page, previous_page=previous_page)
            else:
                prediction = self.model.predict(page=page)
            if prediction.is_first_page:
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
        if self.model.requires_images:
            document.get_images()
        if return_pages:
            processed = self._suggest_first_pages(document, inplace)
        else:
            processed = self._suggest_page_split(document)
        return processed

    def evaluate_full(self, use_training_docs: bool = False, zero_division="warn") -> FileSplittingEvaluation:
        """
        Evaluate the Splitting AI's performance.

        :param use_training_docs: If enabled, runs evaluation on the training data to define its quality; if disabled,
        runs evaluation on the test data.
        :type use_training_docs: bool
        :param zero_division: Defines how to handle situations when precision, recall or F1 measure calculations result
        in zero division.
        Possible values: 'warn' – log a warning and assign a calculated metric a value of 0.
        0 - assign a calculated metric a value of 0.
        'error' – raise a ZeroDivisionError.
        None – assign None to a calculated metric.
        :return: Evaluation information for the model.
        """
        pred_docs = []
        if not use_training_docs:
            original_docs = self.model.test_documents
        else:
            original_docs = self.model.documents
        for doc in original_docs:
            print(f"Processing {doc.id_}.")
            predictions = self.propose_split_documents(doc, return_pages=True)
            assert len(predictions) == 1
            pred_docs.append(predictions[0])
        self.full_evaluation = FileSplittingEvaluation(original_docs, pred_docs, zero_division)
        return self.full_evaluation
