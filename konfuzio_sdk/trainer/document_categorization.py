"""Implements a DocumentModel."""

# import os
import re

# import sys
import logging
from copy import deepcopy
from typing import Union, List
from warnings import warn

# import pathlib

# import cloudpickle

from konfuzio_sdk.data import Project, Document, Category
from konfuzio_sdk.evaluate import CategorizationEvaluation

logger = logging.getLogger(__name__)

warn('This module is WIP: https://gitlab.com/konfuzio/objectives/-/issues/9481', FutureWarning, stacklevel=2)

# ensure backwards compatibility of models saved with project name instead of id.
# Can be deleted when old models are adapted or not needed anymore
NAMES_CONVERSION = {
    'Versicherungspolicen Erweitert (Multipolicen, KFZ, Haftpflicht, ..)': 23,
    'Münchener Verein - Demo': 24,
    'Kontoauszug': 34,
    'Rechnung (German)': 39,
    'BU-Versicherung': 43,
    'RV-Verträge - Demo': 44,
    'KONFUZIO_PAYSLIP_TESTS': 46,
    'ZIMDB-ZIAS: Grundbuchauszug': 50,
    'Arztrechnung': 58,
    'Darlehensvertrag - IDS': 60,
    'Syncier - Financial Statement': 61,
    'ID Cards (from dataset) - Demo': 76,
    'Grundbuchauszug': 145,
    'Kaufvertrag': 169,
    'Expose': 170,
    'Flurkarte': 171,
    'Grundriss': 172,
    'Mietvertrag': 173,
    'Teilungserklaerung': 174,
    'WohnNutzflaechBerech': 175,
}


def get_category_name_for_fallback_prediction(category: Union[Category, str]) -> str:
    """Turn a category name to lowercase, remove parentheses along with their contents, and trim spaces."""
    if isinstance(category, Category):
        category_name = category.name.lower()
    elif isinstance(category, str):
        category_name = category.lower()
    else:
        raise NotImplementedError
    parentheses_removed = re.sub(r'\([^)]*\)', '', category_name).strip()
    single_spaces = parentheses_removed.replace("  ", " ")
    return single_spaces


def build_list_of_relevant_categories(training_categories: List[Category]) -> List[List[str]]:
    """Filter for category name variations which correspond to the given categories, starting from a predefined list."""
    relevant_categories = []
    for training_category in training_categories:
        category_name = get_category_name_for_fallback_prediction(training_category)
        relevant_categories.append(category_name)
    return relevant_categories


class FallbackCategorizationModel:
    """A model that predicts a category for a given document."""

    def __init__(self, project: Union[int, Project], *args, **kwargs):
        """Initialize FallbackCategorizationModel."""
        # Go through keyword arguments, and either save their values to our
        # instance, or raise an error.
        if isinstance(project, int):
            self.project = Project(id_=project)
        elif isinstance(project, Project):
            self.project = project
        else:
            raise NotImplementedError

        self.categories = None
        self.name = self.__class__.__name__

        self.evaluation = None

    def fit(self) -> None:
        """Use as placeholder Function."""
        raise NotImplementedError(
            f'{self} uses a fallback logic for categorizing documents, and does not train a classifier.'
        )

    def save(self, output_dir: str, include_konfuzio=True):
        """Use as placeholder Function."""
        raise NotImplementedError(
            f'{self} uses a fallback logic for categorizing documents, this will not save model to disk.'
        )

    def evaluate(self) -> CategorizationEvaluation:
        """Evaluate the full Categorization pipeline on the pipeline's Test Documents."""
        eval_list = []
        for document in self.test_documents:
            predicted_doc = self.categorize(document=document, recategorize=True)
            eval_list.append((document, predicted_doc))

        self.evaluation = CategorizationEvaluation(self.project, eval_list)

        return self.evaluation
        # raise NotImplementedError(
        #     f'{self} uses a fallback logic for categorizing documents, without using Training or Test documents for '
        #     f'evaluation.'
        # )

    def categorize(self, document: Document, recategorize: bool = False, inplace: bool = False) -> Document:
        """Run categorization."""
        if inplace:
            virtual_doc = document
        else:
            virtual_doc = deepcopy(document)
        if (document.category is not None) and (not recategorize):
            logger.info(
                f'In {document}, the category was already specified as {document.category}, so it wasn\'t categorized '
                f'again. Please use recategorize=True to force running the Categorization AI again on this document.'
            )
            return virtual_doc
        elif recategorize:
            virtual_doc.category = None

        relevant_categories = build_list_of_relevant_categories(self.categories)
        found_category_name = None
        doc_text = virtual_doc.text.lower()
        for candidate_category_name in relevant_categories:
            if candidate_category_name in doc_text:
                found_category_name = candidate_category_name
                break

        if found_category_name is None:
            logger.warning(
                f'{self} could not find the category of {document} by using the fallback logic '
                f'with pre-defined common categories.'
            )
            return virtual_doc
        found_category = [
            category
            for category in self.categories
            if get_category_name_for_fallback_prediction(category) in found_category_name
        ][0]
        virtual_doc.category = found_category
        return virtual_doc


# import numpy as np
# import pandas as pd
# import torch
# import tqdm
# import math
# from PIL import Image
# from torch.utils.data import DataLoader
#
# from konfuzio.classifiers import Classifier
# from konfuzio.default_models import Model
# from konfuzio.default_models.evaluation import show_metrics
# from konfuzio.default_models.utils import build_document_classifier_iterators
# from konfuzio.default_models.utils import build_text_vocab, build_category_vocab
# from konfuzio.default_models.utils import get_document_classifier
# from konfuzio.default_models.utils import df_prediction_as_no_label
# from konfuzio.images_transformations import pre_processing, data_augmentation
# from konfuzio.modules.text import BERT
# from konfuzio.tokenizers import Tokenizer, WhitespaceTokenizer, get_tokenizer
# from konfuzio.vocab import Vocab


class DocumentModel(FallbackCategorizationModel):
    """A model that predicts a category for a given document."""

    def __init__(self, project: Union[int, Project], *args, **kwargs):
        """Initialize a DocumentModel."""
        super().__init__(*args, **kwargs)

    # def __init__(self,
    #              projects: Union[List[Project], None],
    #              tokenizer: Union[Tokenizer, str] = WhitespaceTokenizer(),
    #              image_preprocessing: Union[None, dict] = {'target_size': (1000, 1000), 'grayscale': True},
    #              image_augmentation: Union[None, dict] = {'rotate': 5},
    #              document_classifier_config: dict = {'image_module': {'name': 'efficientnet_b0'},
    #                                                  'text_module': {'name': 'nbowselfattention'},
    #                                                  'multimodal_module': {'name': 'concatenate'}},
    #              text_vocab: Union[None, Vocab] = None,
    #              category_vocab: Union[None, Vocab] = None,
    #              use_cuda: bool = True):
    #     """Initialize a DocumentModel."""
    #     # projects should be a list of at least 2 Projects or None
    #     # where None indicates no training will be done
    #     assert projects is None or (isinstance(projects, List) and
    #                                 all([isinstance(p, Project) for p in projects]))
    #
    #     if projects is not None and len(projects) == 1:
    #         logger.info('You are only using 1 project for the document classification.')
    #
    #     self.projects = projects
    #     self.tokenizer = tokenizer
    #
    #     # if we are using an image module in our classifier then we need to set-up the
    #     # pre-processing and data augmentation for the images
    #     if 'image_module' in document_classifier_config:
    #         self.image_preprocessing = image_preprocessing
    #         self.image_augmentation = image_augmentation
    #         # get preprocessing
    #         preprocessing = pre_processing.ImagePreProcessing(transforms=image_preprocessing)
    #         preprocessing_ops = preprocessing.pre_processing_operations
    #         # get data augmentation
    #         augmentation = data_augmentation.ImageDataAugmentation(transforms=image_augmentation,
    #                                                                pre_processing_operations=preprocessing_ops)
    #         # evaluation transforms are just the preprocessing
    #         # training transforms are the preprocessing + augmentation
    #         self.eval_transforms = preprocessing.get_transforms()
    #         self.train_transforms = augmentation.get_transforms()
    #     else:
    #         # if not using an image module in our classifier then
    #         # our preprocessing and augmentation should be None
    #         assert image_preprocessing is None and image_augmentation is None, \
    #             'If not using an image module then preprocessing/augmentation must be None!'
    #         self.image_preprocessing = None
    #         self.image_augmentation = None
    #         self.eval_transforms = None
    #         self.train_transforms = None
    #
    #     logger.info('setting up vocabs')
    #
    #     # only build a text vocabulary if the classifier has a text module
    #     if 'text_module' in document_classifier_config:
    #         # ensure we have a tokenizer
    #         assert self.tokenizer is not None, 'If using a text module you must pass a Tokenizer!'
    #
    #         if isinstance(self.tokenizer, str):
    #             self.tokenizer = get_tokenizer(tokenizer_name=self.tokenizer, projects=self.projects)
    #
    #         if hasattr(tokenizer, 'vocab'):
    #             # some tokenizers already have a vocab so if they do we use that instead of building one
    #             self.text_vocab = tokenizer.vocab
    #             logger.info('Using tokenizer\'s vocab')
    #         elif text_vocab is None:
    #             # if our classifier has a text module we have a tokenizer that doesn't have a vocab
    #             # then we have to build a vocab from our projects using our tokenizer
    #             self.text_vocab = build_text_vocab(self.projects, self.tokenizer)
    #         else:
    #             self.text_vocab = text_vocab
    #             logger.info('Using provided text vocab')
    #         logger.info(f'Text vocab length: {len(self.text_vocab)}')
    #     else:
    #         # if the classifier doesn't have a text module then we shouldn't have a tokenizer
    #         # and the text vocab should be None
    #         assert tokenizer is None, 'If not using a text module then you should not pass a Tokenizer!'
    #         self.text_vocab = None
    #
    #     # if we do not pass a category vocab then build one
    #     if category_vocab is None:
    #         self.category_vocab = build_category_vocab(self.projects)
    #     else:
    #         self.category_vocab = category_vocab
    #         logger.info('Using provided vocab')
    #
    #     logger.info(f'Category vocab length: {len(self.category_vocab)}')
    #     logger.info(f'Category vocab counts: {self.category_vocab.counter}')
    #
    #     logger.info('setting up document classifier')
    #
    #     # set-up the document classifier
    #     # need to explicitly add input_dim and output_dim as they are calculated from the data
    #     if 'text_module' in document_classifier_config:
    #         document_classifier_config['text_module']['input_dim'] = len(self.text_vocab)
    #     document_classifier_config['output_dim'] = len(self.category_vocab)
    #
    #     # store the classifier config file
    #     self.document_classifier_config = document_classifier_config
    #
    #     # create document classifier from config
    #     self.classifier = get_document_classifier(document_classifier_config)
    #
    #     self.device = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')
    #
    # def save(self, path: Union[None, str] = None, model_type: str = 'DocumentModel') -> None:
    #     """
    #     Save only the necessary parts of the model for extraction/inference.
    #
    #     Saves:
    #     - tokenizer (needed to ensure we tokenize inference examples in the same way that they are trained)
    #     - transforms (to ensure we transform/pre-process images in the same way as training)
    #     - vocabs (to ensure the tokens/labels are mapped to the same integers as training)
    #     - configs (to ensure we load the same models used in training)
    #     - state_dicts (the classifier parameters achived through training)
    #     """
    #     # create dictionary to save all necessary model data
    #     data_to_save = {'tokenizer': self.tokenizer,
    #                     'image_preprocessing': self.image_preprocessing,
    #                     'image_augmentation': self.image_augmentation,
    #                     'text_vocab': self.text_vocab,
    #                     'category_vocab': self.category_vocab,
    #                     'document_classifier_config': self.document_classifier_config,
    #                     'document_classifier_state_dict': self.classifier.state_dict(),
    #                     'model_type': model_type}
    #
    #     path = Model.save(self, data_to_save, path, model_type)
    #
    #     return path
    #
    # def build(self, document_training_config: dict = {}, **kwargs) -> Dict[str, List[float]]:
    #     """Trains the document classifier."""
    #     logger.info('getting document classifier iterators')
    #
    #     # figure out if we need images and/or text depending on if the classifier
    #     # has an image and/or text module
    #     use_image = hasattr(self.classifier, 'image_module')
    #     use_text = hasattr(self.classifier, 'text_module')
    #
    #     if hasattr(self.classifier, 'text_module') and isinstance(self.classifier.text_module, BERT):
    #         document_training_config['max_len'] = self.classifier.text_module.get_max_length()
    #
    #     # get document classifier example iterators
    #     examples = build_document_classifier_iterators(self.projects,
    #                                                    self.tokenizer,
    #                                                    self.eval_transforms,
    #                                                    self.train_transforms,
    #                                                    self.text_vocab,
    #                                                    self.category_vocab,
    #                                                    use_image,
    #                                                    use_text,
    #                                                    **document_training_config,
    #                                                    device=self.device)
    #
    #     train_examples, valid_examples, test_examples = examples
    #
    #     # place document classifier on device
    #     self.classifier = self.classifier.to(self.device)
    #
    #     logger.info('training label classifier')
    #
    #     # fit the document classifier
    #     self.classifier, metrics = self.fit_classifier(train_examples,
    #                                                    valid_examples,
    #                                                    test_examples,
    #                                                    self.classifier,
    #                                                    **document_training_config)
    #
    #     self.evaluate_classifier(test_examples, self.classifier, self.category_vocab)
    #     self.evaluate_classifier_per_document(test_examples, self.classifier, self.category_vocab)
    #
    #     # put document classifier back on cpu to free up GPU memory
    #     self.classifier = self.classifier.to('cpu')
    #
    #     return metrics
    #
    # def get_accuracy(self, predictions: torch.FloatTensor, labels: torch.FloatTensor) -> torch.FloatTensor:
    #     """Calculate accuracy of predictions."""
    #     # predictions = [batch size, n classes]
    #     # labels = [batch size]
    #     batch_size, n_classes = predictions.shape
    #     # which class had the highest probability?
    #     top_predictions = predictions.argmax(dim=1)
    #     # top_predictions = [batch size]
    #     # how many of the highest probability predictions match the label?
    #     correct = top_predictions.eq(labels).sum()
    #     # divide by the batch size to get accuracy per batch
    #     accuracy = correct.float() / batch_size
    #     return accuracy
    #
    # def evaluate_classifier_per_document(self, test_examples: List, classifier: Classifier, prediction_vocab: Vocab):
    #     """Get the predicted and actual classes over the test set."""
    #     predictions, actual_classes, doc_ids = self.predict_documents(test_examples, classifier)
    #
    #     document_id_predictions = dict()
    #     document_id_actual = dict()
    #
    #     for pred, actual, doc_id in zip(predictions, actual_classes, doc_ids):
    #         if str(doc_id) in document_id_predictions.keys():
    #             document_id_predictions[str(doc_id)].append(pred)
    #             document_id_actual[str(doc_id)] = actual
    #         else:
    #             document_id_predictions[str(doc_id)] = [pred]
    #             document_id_actual[str(doc_id)] = actual
    #
    #     predicted_classes = []
    #     actual_classes = []
    #
    #     for doc_id, actual in document_id_actual.items():
    #         page_predictions = torch.stack(document_id_predictions[doc_id])
    #         page_predictions = torch.softmax(page_predictions, dim=-1)  # [n pages, n classes]
    #         mean_page_prediction = page_predictions.mean(dim=0).cpu().numpy()
    #         predicted_class = mean_page_prediction.argmax()
    #
    #         predicted_classes.append(predicted_class)
    #         actual_classes.append(actual)
    #
    #     logger.info('\nResults per document\n')
    #     show_metrics(predicted_classes, actual_classes, prediction_vocab)
    #
    # @torch.no_grad()
    # def predict_documents(self, examples: DataLoader, classifier: Classifier) -> Tuple[List[float], List[float]]:
    #     """Get predictions and true values of the input examples."""
    #     classifier.eval()
    #     actual_classes = []
    #     doc_ids = []
    #     raw_predictions = []
    #
    #     for batch in tqdm.tqdm(examples, desc='Evaluating'):
    #         predictions = classifier(batch)['prediction']
    #         raw_predictions.extend(predictions)
    #         actual_classes.extend(batch['label'].cpu().numpy())
    #         doc_ids.extend(batch['doc_id'].cpu().numpy())
    #     return raw_predictions, actual_classes, doc_ids
    #
    # @torch.no_grad()
    # def extract(self, page_images, text, batch_size=2, *args, **kwargs) -> Tuple[Tuple[str, float], pd.DataFrame]:
    #     """
    #     Get the predicted category for a document.
    #
    #     The document model can have as input the pages text and/or pages images.
    #
    #     The output is a two element Tuple. The first elements contains the category
    #     (category template id or project id)
    #     with maximum confidence predicted by the model and the respective value of confidence (as a Tuple).
    #     The second element is a dataframe with all the categories and the respective confidence values.
    #
    #     category | confidence
    #        A     |     x
    #        B     |     y
    #
    #     In case the model wasn't trained to predict 'NO_LABEL' we can still have it in the output if
    #     the document falls
    #     in any of the following situations.
    #
    #     The output prediction is 'NO_LABEL' if:
    #
    #     - the number of images do not match the number pages text
    #     E.g.: document with 3 pages, 3 images and only 2 pages of text
    #
    #     - empty page text. The output will be nan if it's the only page in the document.
    #     E.g.: blank page
    #
    #     - the model itself predicts it
    #     E.g.: document different from the training data
    #
    #     page_images: images of the document pages
    #     text: document text
    #     batch_size: number of samples for each prediction
    #     :return: tuple of (1) tuple of predicted category and respective confidence nad (2) predictions dataframe
    #     """
    #     # get device and place classifier on device
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     self.classifier = self.classifier.to(device)
    #
    #     # ensure backwards compatibility of models saved with project name instead of id.
    #     # Can be deleted when old models are adapted or not needed anymore
    #     try:
    #         # category_vocab saved with project id
    #         temp_categories = self.category_vocab.get_tokens()
    #         if 'NO_LABEL' in temp_categories:
    #             temp_categories.remove('NO_LABEL')
    #         _ = int(temp_categories[0])
    #         categories = self.category_vocab.get_tokens()
    #     except ValueError:
    #         # category_vocab saved with project name. Conversion is necessary for app compatibility
    #         categories = [str(NAMES_CONVERSION[name]) for name in self.category_vocab.get_tokens()]
    #
    #     # split text into pages
    #     page_text = text.split('\f')
    #
    #     # make sure we have the same number of images as pages of text
    #     if len(page_images) != len(page_text):
    #         logger.error(f'[ERROR] Number of images ({len(page_images)}) '
    #                      f'does not match number of pages ({len(page_text)}.')
    #         predictions_df = df_prediction_as_no_label(categories)
    #         return ('NO_LABEL', 1.0), predictions_df
    #
    #     # does our classifier use text and images?
    #     use_image = hasattr(self.classifier, 'image_module')
    #     use_text = hasattr(self.classifier, 'text_module')
    #
    #     batch_image, batch_text = [], []
    #     predictions = []
    #
    #     try:
    #         # prediction loop
    #         for i, (img, txt) in enumerate(zip(page_images, page_text)):
    #             if use_image:
    #                 # if we are using images, open the image and perform preprocessing
    #                 img = Image.open(img)
    #                 img = self.eval_transforms(img)
    #                 batch_image.append(img)
    #             if use_text:
    #                 # if we are using text, tokenize and numericalize the text
    #                 if isinstance(self.classifier.text_module, BERT):
    #                     max_length = self.classifier.text_module.get_max_length()
    #                 else:
    #                     max_length = None
    #                 tok = self.tokenizer.get_tokens(txt)[:max_length]
    #                 # assert we have a valid token (e.g '\n\n\n' results in tok = [])
    #                 if len(tok) <= 0:
    #                     logger.info(f'[WARNING] The token resultant from page {i} is empty. Page text: {txt}.')
    #
    #                 idx = [self.text_vocab.stoi(t) for t in tok]
    #                 txt_coded = torch.LongTensor(idx)
    #                 batch_text.append(txt_coded)
    #             # need to use an `or` here as we might not be using one of images or text
    #             if len(batch_image) >= batch_size or len(batch_text) >= batch_size or i == (len(page_images) - 1):
    #                 # create the batch and get prediction per page
    #                 batch = {}
    #                 if use_image:
    #                     batch['image'] = torch.stack(batch_image).to(device)
    #                 if use_text:
    #                     batch_text = torch.nn.utils.rnn.pad_sequence(batch_text, batch_first=True,
    #                                                                  padding_value=self.text_vocab.pad_idx)
    #                     batch['text'] = batch_text.to(device)
    #
    #                 if use_text and batch['text'].size()[1] == 0:
    #                     # There is no text in the batch. Text is empty (page token not valid).
    #                     # If using a Bert model, the prediction will fail. We skip the prediction and add nan instead.
    #                     prediction = torch.tensor([[np.nan for _ in range(len(self.category_vocab))]]).to(device)
    #                 else:
    #                     prediction = self.classifier(batch)['prediction']
    #
    #                 predictions.extend(prediction)
    #                 batch_image, batch_text = [], []
    #     except Exception:
    #         logger.error('[ERROR] Unexpected error in the prediction loop of DocumentModel.')
    #         predictions_df = df_prediction_as_no_label(categories)
    #         return ('NO_LABEL', 1.0), predictions_df
    #
    #     try:
    #         # stack prediction per page, use softmax to convert to probability and average across
    #         predictions = torch.stack(predictions)  # [n pages, n classes]
    #
    #         if predictions.shape != (len(page_images), len(self.category_vocab)):
    #             logger.error(f'[ERROR] Predictions shape {predictions.shape} different '
    #                          f'than expected {(len(page_images), len(self.category_vocab))}')
    #         predictions = torch.softmax(predictions, dim=-1).cpu().numpy()  # [n pages, n classes]
    #
    #         # remove invalid pages
    #         predictions_filtered = [p for p in predictions if not all(np.isnan(x) for x in p)]
    #
    #         if len(predictions_filtered) > 0:
    #             mean_prediction = np.array(predictions_filtered).mean(axis=0)  # [n classes]
    #
    #         else:
    #             # All predictions were nan
    #             # We can have all predictions as nan if, for example, the document is a blank page.
    #             logger.info('[WARNING] All predictions are nan.')
    #             predictions_df = df_prediction_as_no_label(categories)
    #             return ('NO_LABEL', 1.0), predictions_df
    #
    #         # differences might happen due to floating points numerical errors
    #         if not math.isclose(sum(mean_prediction), 1.0, abs_tol=1e-4):
    #             logger.error(f'[ERROR] Sum of the predictions ({sum(mean_prediction)}) is not 1.0.')
    #
    #     except Exception:
    #         logger.error('[ERROR] Unexpected error in the prediction processing of DocumentModel.')
    #         predictions_df = df_prediction_as_no_label(categories)
    #         return ('NO_LABEL', 1.0), predictions_df
    #
    #     category_preds = dict()
    #
    #     # store the prediction confidence per label
    #     for idx, label in enumerate(categories):
    #         category_preds[label] = mean_prediction[idx]
    #
    #     # store prediction confidences in a df
    #     predictions_df = pd.DataFrame(data={'category': list(category_preds.keys()),
    #                                         'confidence': list(category_preds.values())})
    #
    #     # which class did we predict?
    #     # what was the label of that class?
    #     # what was the confidence of that class?
    #     predicted_class = int(mean_prediction.argmax())
    #     predicted_label = categories[predicted_class]
    #     predicted_confidence = mean_prediction[predicted_class]
    #
    #     return (predicted_label, predicted_confidence), predictions_df
    #
    #
