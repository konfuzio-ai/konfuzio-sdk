import collections
import logging
import os
import time
from functools import partial

import pandas as pd
import numpy as np

from konfuzio_sdk.evaluate import Evaluation
from konfuzio_sdk.tokenizer.base import ListTokenizer

from pathos.multiprocessing import ProcessPool
from typing import List, Tuple, Dict, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, \
    confusion_matrix, precision_score
from tabulate import tabulate

from konfuzio_sdk.pipelines.features import process_document_data, get_span_features, get_y_train, convert_to_feat
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer, WhitespaceNoPunctuationTokenizer, ConnectedTextTokenizer, \
    ColonPrecededTokenizer, CapitalizedTextTokenizer, NonTextTokenizer, RegexMatcherTokenizer

from konfuzio_sdk.data import Category, Document, Annotation

from konfuzio_sdk.pipelines.base import ExtractionModel
from konfuzio_sdk.utils import get_timestamp

logger = logging.getLogger(__name__)


class DocumentAnnotationMultiClassModel(ExtractionModel):
    """
    Create an Extraction AI capable of extracting Annotations with a Label and Label Set classification.

    Both Label and Label Set classifiers are using a RandomForestClassifier from scikit-learn.
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    A random forest classifier is a group of decision trees classifiers.

    The parameters of this class allow to select the Tokenizer, to configure the Label and Label Set classifiers and to
    select the type of features used by the Label and Label Set classifiers.

    They are divided in:
    - tokenizer selection
    - parametrization of the Label classifier
    - parametrization of the Label Set classifier
    - features for the Label classifier
    - features for the Label Set classifier

    By default, the text of the Documents is split into smaller chunks of text based on whitespaces
    ('tokenizer_whitespace'). That means that all words present in the text will be shown to the AI. It is possible to
    define if the splitting of the text into smaller chunks should be done based on regexes learned from the
    Spans of the Annotations of the Category ('tokenizer_regex') or if to use a model from Spacy library for German
    language ('tokenizer_spacy'). Another option is to use a pre-defined list of tokenizers based on regexes
    ('tokenizer_regex_list') and, on top of the pre-defined list, to create tokenizers that match what is missed
    by those ('tokenizer_regex_combination').

    Some parameters of the scikit-learn RandomForestClassifier used for the Label and/or Label Set classifier
    can be set directly in Konfuzio Server ('label_n_estimators', 'label_max_depth', 'label_class_weight',
    'label_random_state', 'label_set_n_estimators', 'label_set_max_depth').

    Features are measurable pieces of data of the Annotation. By default, a combination of features is used that
    includes features built from the text of the Annotation ('string_features'), features built from the position of
    the Annotation in the Document ('spatial_features') and features from the Spans created by a WhitespaceTokenizer on
    the left or on the right of the Annotation ('n_nearest_left', 'n_nearest_right', 'n_nearest_across_lines).
    It is possible to exclude any of them ('spatial_features', 'string_features', 'n_nearest_left', 'n_nearest_right')
    or to specify the number of Spans created by a WhitespaceTokenizer to consider
    ('n_nearest_left', 'n_nearest_right').

    While extracting, the Label Set classifier takes the predictions from the Label classifier as input.
    The Label Set classifier groups them intoAnnotation sets.
    It is possible to define the confidence threshold for the predictions to be considered by the
    Label Set classifier ('label_set_confidence_threshold'). However, the label_set_confidence_threshold is not applied
    to the final predictions of the Extraction AI.

    Parameters
    ----------
    tokenizer_whitespace : bool, default=False
        Whether the text of the Documents is split based on whitespaces by the WhitespaceTokenizer.

    tokenizer_regex : bool, default=False
        Whether the text of the Documents is split by the RegexTokenizer that learns to do the splitting based on the
        Spans of the Annotations in the Category. It is based on regexes.

    tokenizer_regex_list : bool, default=False
        Whether the text of the Documents is split by a predefined combination of RegexTokenizers. This combination
        splits the text based on regexes that identify common patterns such as capitalized entities.

    tokenizer_regex_combination : bool, default=False
        Whether the text of the Documents is split by a predefined combination of RegexTokenizers plus RegexTokenizers
        built automatically to match what is missed by the predefined ones.

    tokenizer_spacy : bool, default=False
        Whether the text of the Documents is split by the SpacyTokenizer that uses the Spacy model 'de_core_news_sm'
        for German language.

    label_n_estimators : int, default=100
        Number of trees in the forest (see RandomForestClassifier from scikit-learn).

    label_max_depth : int, default=100
        Maximum depth of the tree (see RandomForestClassifier from scikit-learn).

    label_class_weight : str, default=“balanced”
        Adjust the weights for the classes accordingly with its representation in the dataset (“balanced”,
        “balanced_subsample”) (see RandomForestClassifier from scikit-learn).

    label_random_state : int, default=420
        Controls both the randomness of the bootstrapping of the samples and the sampling of the features (see
        RandomForestClassifier from scikit-learn).

    label_set_n_estimators : int, default=100
        Number of trees in the forest (see RandomForestClassifier from scikit-learn).

    label_set_max_depth : int, default=100
        Maximum depth of the tree (see RandomForestClassifier from scikit-learn).

    spatial_features : bool, default=True
        Whether features based on the position of the Annotation in the Document(e.g. coordinates) are used.

    string_features : bool, default=True
        Whether features from the text of the Annotation (e.g. number of digits) are used.

    n_nearest_left : int, default=2
        Number of Spans created by a WhitespaceTokenizer to use as features on the left side of an Annotation.

    n_nearest_right : int, default=2
        Number of Spans created by a WhitespaceTokenizer to use as features on the right side of an Annotation.

    n_nearest_across_lines : bool, default=True
        Whether previous and subsequent lines are considered when creating left and right Spans with
        the WhitespaceTokenizer.

    label_set_confidence_threshold : float, default=0.5
        Threshold for the confidence of the Label classifier prediction so that it's considered by the Label Set
        classifier.

    Notes
    -----
        There are constants that represent decisions made for this Extraction AI and cannot be modified.
        They further specify the Label or Label Set classifier training, how missing values are handled and
        how the metrics are calculated.

    _LABEL_TARGET_COLUMN_NAME : str, default='label_text'
        Name of the target column for the training of the Label classifier.

    _META_INFORMATION_LIST : list
        Meta information that is always included in the Spans' dataframe of the Label classifier.

    _ABS_POS_FEATURE_LIST : list
        Name of the positional features to be used for training the Label classifier.
        They are retrieved directly from Span and are used if spatial_features is True.

    _DEFAULT_NEIGHBOUR_DIST : int, default=10000
        Default value for the distance to the next Span in case no Span is present.
        It is used in the features from the Spans created by a WhitespaceTokenizer around the Annotation.

    _DEFAULT_NEIGHBOUR_STRING: str, default=''
        Default offset_string of the next Span in case no Span is present.
        It is used in the features from the Spans created by a WhitespaceTokenizer around the Annotation.

    _GENERAL_RESULTS_AVERAGE_MODE : str, default='weighted'
        Defines how the predicted classes are averaged when calculating f1_score for the Label classifier.

    _LABEL_SET_TARGET_COLUMN_NAME : str, default='y'
        Name of the column with the target for the training of the Label Set classifier.

    _LABEL_SET_TARGET_DEFAULT_VALUE : str, default='No'
        Default value for the target column of the Label Set Classifier.

    _DEFAULT_VALUE_MISSING_LABEL : int, default=0
        Default value for Labels missing in the training dataframe of the Label Set Classifier.

    _LABEL_SET_FEATURE_DEFAULT_VALUE : int, default=0
        Default value for features in the training dataframe of the Label Set Classifier.

    _LABEL_SET_RESULTS_AVERAGE_MODE : str, default='micro'
        Defines how the predicted classes are averaged when calculating precision and recall for
        the Label Set classifier.

    _PROBABILITY_DISTRIBUTION_STEP : float, default=0.1
        Starting value for the confidence range in the probability distribution calculated after training.

    _PROBABILITY_DISTRIBUTION_START : float, default=0.0
        Step size for the confidence intervals in the probability distribution calculated after training.
    """
    # Tokenizer selection
    tokenizer_whitespace: bool = False
    tokenizer_regex: bool = False
    tokenizer_regex_list: bool = False
    tokenizer_regex_combination: bool = False
    tokenizer_spacy: bool = False

    # Parametrization configs for Label classifier
    label_n_estimators: int = 100
    label_max_depth: int = 100
    label_class_weight: str = 'balanced'
    label_random_state: int = 420

    # Parametrization configs for Label Set classifier
    label_set_n_estimators: int = 100
    label_set_max_depth: int = 100

    # Features configs for Label classifier
    spatial_features: bool = True
    string_features: bool = True
    n_nearest_across_lines: bool = True
    n_nearest_left: int = 2
    n_nearest_right: int = 2
    # first_word: list = []  # Not implemented
    # substring_features: list = []  # Not implemented
    # catchphrase_features: list = []  # Not implemented

    # Features configs for Label Set classifier
    # n_nearest_label_set: int = 5  # Not implemented
    # Confidence threshold for Label Set (filter label predictions bellow threshold)
    label_set_confidence_threshold: float = 0.5

    # Training configs (both classifiers)
    # train_split_percentage: float = 1.0
    # multiprocessing: bool = False  # Not used atm
    # separate_labels: bool = False  # Not used atm

    # Constants

    # Name of the target column for the training of the Label classifier.
    _LABEL_TARGET_COLUMN_NAME: str = 'label_text'
    # Meta information that is always included in the Spans' dataframe of the Label classifier.
    _META_INFORMATION_LIST: list = [
        'annotation_id',
        # 'bottom',
        'confidence',
        'document_id',
        'end_offset',
        'id_',
        'is_correct',
        'normalized',
        'offset_string',
        'revised',
        'start_offset',
        'line_index',
        # 'top',
    ]
    # TODO: constants (decisions are made) - make sure they cannot be overwritten
    #   define as property?
    # Name of the positional features to be used for training the Label classifier.
    _ABS_POS_FEATURE_LIST: list = [
        "x0",
        "y0",
        "x1",
        "y1",
        "x0_relative",
        "x1_relative",
        "y0_relative",
        "y1_relative",
        "page_index",
        "page_index_relative",
    ]
    # TODO what is fallback value for no neighbour? how does this value influence feature scaling?
    # Default value for the distance to the next Span in case no Span is present.
    _DEFAULT_NEIGHBOUR_DIST: int = 10000
    # Default offset_string of the next Span in case no Span is present.
    _DEFAULT_NEIGHBOUR_STRING: str = ''
    # Defines how the predicted classes are averaged when calculating f1_score for the Label classifier.
    _GENERAL_RESULTS_AVERAGE_MODE: str = 'weighted'

    # Name of the column with the target for the training of the Label Set classifier.
    _LABEL_SET_TARGET_COLUMN_NAME: str = 'y'
    # Default value for the target column of the Label Set Classifier.
    _LABEL_SET_TARGET_DEFAULT_VALUE: str = 'No'
    # Default value for Labels missing in the training dataframe of the Label Set Classifier.
    _DEFAULT_VALUE_MISSING_LABEL: int = 0
    # Default value for features in the training dataframe of the Label Set Classifier.
    _LABEL_SET_FEATURE_DEFAULT_VALUE: int = 0
    # Defines how the predicted classes are averaged when calculating precision and recall for
    #         the Label Set classifier.
    _LABEL_SET_RESULTS_AVERAGE_MODE: str = 'micro'

    # Probability distribution
    # Starting value for the confidence range in the probability distribution calculated after training.
    _PROBABILITY_DISTRIBUTION_STEP: float = 0.1
    # Step size for the confidence intervals in the probability distribution calculated after training.
    _PROBABILITY_DISTRIBUTION_START: float = 0.0

    def __init__(self, category: Category):
        """DocumentAnnotationModel."""
        super().__init__()

        # Set category
        self.category = category

        # Set tokenizer
        self.tokenizer = None

        # Set training documents
        self.documents = self.category.documents()

        # Set test documents
        self.test_documents = self.category.test_documents()

        # Set Labels and Label Sets
        self.label_sets: List['LabelSet'] = self.category.label_sets
        # TODO how can we get all labels without using list(set(x))?
        self.labels: List['Label'] = list(set(label for label_set in self.label_sets for label in label_set.labels))

        # Empty Label and Label Set to create the potential Annotations
        self.no_label = self.category.project.no_label
        self.no_label_set = self.category.project.no_label_set

        # Set Label Set classifier
        self.label_set_clf = None
        self.label_set_feature_list = None

    @classmethod
    def get_param_names(cls):
        """Get params that can be configured in parameter search."""
        param_names = {
            value for key, value in cls.__dict__.items()
            if not key.startswith('__') and not key.startswith('_') and not callable(value)
        }
        return param_names

    def get_params(self):
        # get model parameters and their configuration in the current training iteration
        model_parameters = {
            key: self.__getattribute__(key) for key in dir(self) if key in self.get_param_names()
        }
        return model_parameters

    def configure(self, config):
        """Configure model parameters."""
        for key, value in config.items():
            if key.startswith('_'):
                raise ValueError(f'{key} cannot be configured because is a constant of the class.')
            if isinstance(value, type(getattr(self, key))):
                setattr(self, key, value)
            else:
                raise ValueError
        return self

    def data_checks(self):
        """Check data consistency."""
        for document in self.category.documents() + self.category.test_documents():
            if not document.check_bbox():
                print(f'Document {document} is invalid, see logs.')

    def build(self):
        """Build an DocumentAnnotationMultiClassModel."""
        logger.info('Start data_checks()...')
        self.data_checks()
        logger.info('Start fit_tokenizer()...')
        self.fit_tokenizer()
        logger.info('Start tokenize()...')
        self.tokenize(documents=self.documents + self.test_documents)
        logger.info('Start create_candidates_dataset()...')
        self.create_candidates_dataset()
        logger.info('Start train_valid_split()...')
        self.train_valid_split()
        logger.info('Start fit()...')
        self.fit()
        logger.info('Start fit_label_set_clf()...')
        self.fit_label_set_clf()
        logger.info('Start evaluate()...')
        self.evaluate()

        return self

    # the extract function must accept arbitrary args and kwargs to support multiple server versions.
    def extract(self, text: str, bbox: dict, pages: list, *args, **kwargs) -> 'Dict':
        """
        Use self to extract all known labels and label_sets.

        :param text: HTML or raw text of document
        :param bbox: Bbox of the document
        :param pages: The dimension of the pages of the document
        :return: dictionary of labels and top candidates

        """
        document = Document(
            text=text,
            bbox=bbox,
            project=self.category.project,
            category=self.category,
            pages=pages,
        )

        # TODO: add these verifications as a step in build()
        # document must have text
        if document.text is None or not document.text.strip():
            # text is None or only has whitespaces
            logger.warning(f'Document {document} cannot be extracted because does not contain valid text. Text: {text}')
            return {}

        # document must have bbox if we are using spatial features or features from the Spans created by a
        # WhitespaceTokenizer on the left or on the right of the Annotation
        if document.get_bbox() is None or document.get_bbox() == {} and \
                (self.spatial_features or self.n_nearest_left > 0 or self.n_nearest_right > 0):
            # bbox is None or empty
            logger.warning(f'Document {document} cannot be extracted because does not contain valid bbox. Bbox: {bbox}')
            return {}

        # No Annotations must be loaded
        assert document._annotations is None

        self.tokenize([document], multiprocess=False)

        # Generate features
        df, _ = self.feature_function(documents=[document], multiprocess=False)
        if df.empty:
            return {}

        # if self.clf is None and hasattr(self, 'label_clf'):  # Can be removed for models after 09.10.2020
        #    self.clf = self.label_clf

        results = pd.DataFrame(
            data=self.clf.predict_proba(X=df[self.label_feature_list]),
            columns=self.clf.classes_
        )

        # Remove no_label predictions
        # TODO:
        # There might be a difference between:
        # (1) 70% No_label, 30% Firstname.
        # (2) 30% Firstname, 25%..., 20%....10%, 8%
        # at the moment its the same.
        if self.no_label.name in results.columns:
            results = results.drop([self.no_label.name], axis=1)

        # Store most likely prediction and its accuracy in separated columns
        df[self._LABEL_TARGET_COLUMN_NAME] = results.idxmax(axis=1)
        df['label_id'] = df[self._LABEL_TARGET_COLUMN_NAME].replace(dict((x.name, x.id_) for x in self.labels))
        df['confidence'] = results.max(axis=1)

        missing_features = ['r_offset_string1', 'r_offset_string0', 'l_offset_string0', 'l_offset_string1']
        if missing_features in df.columns.to_list():
            drop_columns = self.label_feature_list + missing_features
        else:
            drop_columns = self.label_feature_list
        # TODO:  missing_features are missing in features list
        df = df.drop(drop_columns, axis=1)
        # extract_threshold = None  # TODO make this a parameter?
        # However not filtering has the advantage to check the quality of the tokenizer.
        # if extract_threshold is not None:
        #   df = df[df['confidence'] > extract_threshold]

        # Convert DataFrame to Dict with labels as keys and label dataframes as value.
        # TODO multiple conversion. Convert only once, or better never.
        res_dict = {}
        for label_text in set(df[self._LABEL_TARGET_COLUMN_NAME]):
            label_df = df[df[self._LABEL_TARGET_COLUMN_NAME] == label_text].copy()
            if not label_df.empty:
                res_dict[label_text] = label_df

        # # TODO do processing per Page, or in other words on which level can we parallelize (e.g. Spans)
        #
        # # Filter results that are bellow the extract threshold
        # # (helpful to reduce the size in case of many predictions/ big documents)
        # if hasattr(self, 'extract_threshold') and self.extract_threshold is not None:  # TODO no hasattr
        #     logger.info('Filtering res_dict')
        #     for label, value in res_dict.items():
        #         if isinstance(value, pd.DataFrame):
        #             res_dict[label] = value[value['confidence'] > self.extract_threshold]

        # Try to calculate annotation_sets based on label_set classifier.

        res_dict = self.extract_label_set_with_clf(document, df, res_dict)

        return res_dict

    def create_candidates_dataset(self):
        """
        Create df_train, df_test and self.label_feature_list based on the feature function.

        :return:
        """
        self.df_train, self.label_feature_list = self.feature_function(documents=self.documents)

        if self.df_train.empty:
            logger.warning('df_train is empty! No training data found.')
            return None

        self.df_test, test_label_feature_list = self.feature_function(documents=self.test_documents)

        if not self.df_test.empty:
            assert self.label_feature_list == test_label_feature_list

        return self

    def fit_tokenizer(self):
        # Tokenizer selection
        tokenizer_verification = [
            self.tokenizer_whitespace + self.tokenizer_regex + self.tokenizer_spacy + self.tokenizer_regex_list +
            self.tokenizer_regex_combination
        ]
        if sum(tokenizer_verification) > 1:
            raise ValueError(f'You can only select 1 tokenizer. You have White space {self.tokenizer_whitespace}, '
                             f'Regex {self.tokenizer_regex}, Regex list {self.tokenizer_regex_list},'
                             f'Regex combination: {self.tokenizer_regex_combination}, Spacy: {self.tokenizer_spacy}.')

        if self.tokenizer_whitespace:
            self.tokenizer = WhitespaceTokenizer()

        elif self.tokenizer_regex_list:
            self.tokenizer = ListTokenizer(
                tokenizers=[
                    WhitespaceTokenizer(),
                    WhitespaceNoPunctuationTokenizer(),
                    ConnectedTextTokenizer(),
                    ColonPrecededTokenizer(),
                    CapitalizedTextTokenizer(),
                    NonTextTokenizer(),
                ]
            )

        elif self.tokenizer_regex_combination:
            self.tokenizer = RegexMatcherTokenizer(
                tokenizers=[
                    WhitespaceTokenizer(),
                    WhitespaceNoPunctuationTokenizer(),
                    ConnectedTextTokenizer(),
                    ColonPrecededTokenizer(),
                    CapitalizedTextTokenizer(),
                    NonTextTokenizer(),
                ]
            )
            self.tokenizer.fit(category=self.category)

        else:
            self.tokenizer = WhitespaceTokenizer()
            logger.info('Using WhitespaceTokenizer by default.')

    def tokenize(self, documents: List['Document'], multiprocess=True):
        """Call the tokenizer on test and training documents."""

        if documents is None:
            documents = self.category.documents() + self.category.test_documents()

        if self.tokenizer is None:
            raise ValueError(f'Tokenizer not defined.')

        def _tokenize(document):
            document.annotations()
            try:
                self.tokenizer.tokenize(document, multiprocess=False)
            except ValueError:  # It is OK if NO_Label annotations cannot be added.
                pass

        if multiprocess:
            pool = ProcessPool()
            pool.map(_tokenize, documents)
        else:
            for document in documents:
                _tokenize(document)

        #         # todo go back to stable approach one
        #         with ProcessPool() as p:
        #             # we use pathos instead of the normal multiprocess python package
        #             # because the generic candidate function cannot be serialized by pickle
        #             # error:  multiprocessing.pool.MaybeEncodingError: Error sending result:
        #             # '[<project_label_set.main.BuildingType object at 0x7f60eabe0588>]'.
        #             # Reason: 'AttributeError("Can't pickle local object
        #             # 'generic_candidate_function.<locals>.candidate_function'",)'
        #
        #             # see https://stackoverflow.com/a/21345308/11680891
        #             self.labels = p.amap(build_my_model, self.labels)
        #             self.categories = p.amap(build_my_model, self.categories)

        # for document in documents:
        #     # Load Annotations before doing tokenization.
        #
        #
        #     # # Load bboxes for all spans
        #     # for annotation in document.annotations():
        #     #     for span in annotation.spans:
        #     #         span.bbox()
        #     #
        #     # # todo remove
        #     # if len([s for a in document.annotations() for s in a.spans if s.x0 is None or s.y0 is None]):
        #     #     raise Exception('Bbox not set.')
        #     # until here

    def lose_weight(self):
        """Lose weight before pickling."""
        super().lose_weight()

        self.category.project.lose_weight()

        for label in self.labels:
            label.lose_weight()
        for label_set in self.label_sets:
            label_set.lose_weight()

        # remove documents
        self.documents = []
        self.test_documents = []

        # remove processing steps of Tokenizer
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, 'lose_weight'):  # remove after merge of branch tokenizer from SDK 03-05-2022
                self.tokenizer.lose_weight()

    def feature_function(self, documents, multiprocess=False) -> Tuple[pd.DataFrame, list]:
        """Return a df with all the data from the json files read out and properly converted."""
        logger.info('Start generating features.')

        if not documents:
            df = pd.DataFrame()
            return df, []

        if self.spatial_features:
            abs_pos_feature_list = self._ABS_POS_FEATURE_LIST
        else:
            abs_pos_feature_list = []

        def _feature_function(document):
            document_features = []
            document_feature_list = []
            # training_annotations = document.annotations(use_correct=True)

            # TODO duplication should not be needed anymore
            # annotations_labeled = self._filter_annotations_for_duplicates(annotations_labeled)

            # no label annotations have a label with id_ None
            no_label_annotations = document.annotations(use_correct=False, label=self.no_label)
            label_annotations = [x for x in document.annotations(use_correct=True) if x.label != self.no_label]
            # del annotations_labeled  # why do we need this in the first place.

            logger.info(f'Document {document} has {len(label_annotations)} labeled annotations')
            logger.info(f'Document {document} has {len(no_label_annotations)} NO_LABEL annotations')

            training_annotations = label_annotations + no_label_annotations

            # no_label_limit deactivated
            # if isinstance(no_label_limit, int):
            #     n_no_labels = no_label_limit
            # elif isinstance(no_label_limit, float):
            #     n_no_labels = int(len(label_annotations) * no_label_limit)
            # else:
            #     assert no_label_limit is None
            #
            # if no_label_limit is not None:
            #     no_label_annotations = self.get_best_no_label_annotations(n_no_labels,
            #                                                               label_annotations,
            #                                                               no_label_annotations)
            #     logger.info(f'Document {document} NO_LABEL annotations has been reduced to {len(no_label_annotations)}')
            # training_annotations = sorted(label_annotations + no_label_annotations)

            t0 = time.monotonic()

            if self.n_nearest_left > 0 or self.n_nearest_right > 0:
                # TODO: other functions that create features are defined in konfuzio.multiclass_clf
                n_nearest_df, n_nearest_feature_list = self.get_n_nearest_features(
                    document=document,
                    annotations=training_annotations,
                )
                document_feature_list += n_nearest_feature_list
                document_features.append(n_nearest_df)

            if self.string_features:
                string_feature_df, string_feature_list = process_document_data(
                    document=document,
                    annotations=training_annotations,
                )
                document_feature_list += string_feature_list
                document_features.append(string_feature_df)

            span_feature_df, span_feature_list = get_span_features(
                annotations=training_annotations,
                abs_pos_feature_list=abs_pos_feature_list,
                meta_information_list=self._META_INFORMATION_LIST,
            )
            document_feature_list += span_feature_list
            document_features.append(span_feature_df)

            document_df = pd.concat(document_features, axis=1)
            # Set label_text
            document_df[self._LABEL_TARGET_COLUMN_NAME] = get_y_train(training_annotations)

            dt = time.monotonic() - t0
            logger.info(f'Document {document} processed in {dt:.1f} seconds.')

            if len(list(set(document_feature_list))) != len(document_feature_list):
                raise ValueError('Duplicated features detected.')

            return document_feature_list, document_df

        if multiprocess:
            pool = ProcessPool()
            feature_list, document_df_list = zip(*pool.map(_feature_function, documents))
        else:
            result = [_feature_function(document) for document in documents]
            feature_list, document_df_list = zip(*result)

        df = pd.concat(document_df_list).reset_index(drop=True)

        if not df.empty:
            if df[self._LABEL_TARGET_COLUMN_NAME].isnull().sum() > 0:
                raise Exception(f'NaN value in {self._LABEL_TARGET_COLUMN_NAME} column')

        # TODO outlier?

        # Feature Scaling TODO should we use this?
        # from sklearn.preprocessing import StandardScaler
        #
        # sc = StandardScaler()
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.transform(X_test)

        # Check that all documents have the same feature.
        assert all(feature_list[0] == x for x in feature_list)

        return df, feature_list[0]

    # Unclear hwo its works when tokenizer is no WhiteSpaceTokenizer (e.g. it creates overlapping annotations)
    def get_n_nearest_features(
            self,
            document: Document,
            annotations: List[Annotation],
    ):
        # TODO: Feels wrong to tokenize again and to create a new virtual doc.
        # How should "neighbour" work when we use a tokenizer different than WhiteSpacetokenizer?
        tokenizer = WhitespaceTokenizer()
        document._bbox = document.get_bbox()
        virtual_doc = Document(
            text=document.text,
            bbox=document.get_bbox(),
            project=document.project,
            category=document.category,
            pages=document.pages
        )
        virtual_doc._bbox = document.get_bbox()
        tokenizer.tokenize(virtual_doc)
        whitespace_spans = [span for a in virtual_doc.annotations(use_correct=False) for span in a.spans]

        # TODO: Should these variables be defined in the init?
        left_prefix = 'l_'
        right_prefix = 'r_'
        distance_name = 'dist'
        offset_string_name = 'offset_string'

        # l_dist + r_dist
        l_keys = [left_prefix + distance_name + str(x) for x in range(self.n_nearest_left)]
        r_keys = [right_prefix + distance_name + str(x) for x in range(self.n_nearest_right)]

        features_list: List[Dict] = []
        relative_string_feature_list: List[str] = []
        relative_string_feature_list += l_keys
        relative_string_feature_list += r_keys

        # TODO: Should be a SDK concept?
        for annotation in annotations:
            for span in annotation.spans:
                span.bbox()
                feature_dict = {}
                l_indexes = [(look_left_index, -1, left_prefix) for look_left_index in range(0, self.n_nearest_left)]
                r_indexes = [(look_right_index, 1, right_prefix) for look_right_index in range(0, self.n_nearest_right)]
                for index, factor, prefix in l_indexes + r_indexes:
                    try:
                        current_index = whitespace_spans.index(span)
                    except ValueError:
                       current_index = sorted(whitespace_spans + [span]).index(span) + factor * 1

                    span_index = current_index + factor * (index + 1)
                    if 0 <= span_index < len(whitespace_spans):
                        lr_span = whitespace_spans[span_index]  # look-left index is 0-based
                    else:
                        lr_span = None

                    if lr_span and (self.n_nearest_across_lines or lr_span.line_index == span.line_index):
                        lr_span.bbox()
                        feature_dict[prefix + distance_name + str(index)] = lr_span.x0 - lr_span.x1
                        feature_dict[prefix + offset_string_name + str(index)] = lr_span.offset_string
                    else:
                        # TODO what is fallback value for no neighbour?, how does this value influence feature scaling?
                        feature_dict[prefix + distance_name + str(index)] = self._DEFAULT_NEIGHBOUR_DIST
                        # https://stackoverflow.com/questions/58971596/random-forest-make-null-values-always-have-their-own-branch-in-a-decision-tree
                        feature_dict[prefix + offset_string_name + str(index)] = self._DEFAULT_NEIGHBOUR_STRING
                features_list.append(feature_dict)
        if not annotations:
            return pd.DataFrame(), []

        df = pd.DataFrame(features_list)

        if not annotations:
            logger.error('If there are no annotations the next loop will crash.')

        for index, factor, prefix in l_indexes + r_indexes:
            df_string_features = convert_to_feat(
                list(df[prefix + offset_string_name + str(index)]), prefix[0] + str(index) + '_'
            )
            relative_string_feature_list += list(df_string_features.columns.values)
            df = df.join(df_string_features, lsuffix='_caller', rsuffix='_other')

        return df, relative_string_feature_list

    # def get_best_no_label_annotations(self, n_no_labels: int, label_annotations: List[Annotation],
    #                                   no_label_annotations: List[Annotation]) -> List[Annotation]:
    #     """Select no_label annotations which are probably most beneficial for training."""
    #     # store our chosen "best" NO_LABELS
    #     best_no_label_annotations = []
    #
    #     # get all the real label offset strings and offsets
    #     label_texts = set([a.offset_string for a in label_annotations])
    #     offsets = set([(a.start_offset, a.end_offset) for a in label_annotations])
    #
    #     _no_label_annotations = []
    #
    #     random.shuffle(no_label_annotations)
    #
    #     # for every NO_LABEL that has an exact string match (but not an offset match)
    #     # to a real label, we add it to the best_no_label_annotations
    #     for annotation in no_label_annotations:
    #         # TODO: start_offset/ end offset on Annotation Level is legacy #8820 #8817
    #         offset_string = annotation.offset_string
    #         start_offset = annotation.start_offset
    #         end_offset = annotation.end_offset
    #         if offset_string in label_texts and (start_offset, end_offset) not in offsets:
    #             best_no_label_annotations.append(annotation)
    #         else:
    #             _no_label_annotations.append(annotation)
    #
    #     # if we have enough NO_LABELS, we stop here
    #     if len(best_no_label_annotations) >= n_no_labels:
    #         return best_no_label_annotations[:n_no_labels]
    #
    #     no_label_annotations = _no_label_annotations
    #     _no_label_annotations = collections.defaultdict(list)
    #
    #     # if we didn't have enough exact matches then we want our NO_LABELS to be the same
    #     # data_type as our real labels
    #     # we count the amount of each data_type in the real labels
    #     # then calculate how many NO_LABEL of each data_type we need
    #     data_type_count = collections.Counter()
    #     data_type_count.update([a.label.data_type for a in label_annotations])
    #     for data_type, count in data_type_count.items():
    #         data_type_count[data_type] = n_no_labels * count / len(label_annotations)
    #
    #     random.shuffle(no_label_annotations)
    #
    #     # we now loop through the NO_LABELS that weren't exact matches and add them to
    #     # the _no_label_annotations dict if we still need more of that data_type
    #     # any that belong to a different data_type are added under the 'extra' key
    #     for annotation in no_label_annotations:
    #         data_type = self.predict_data_type(annotation)
    #         if data_type in data_type_count:
    #             if len(_no_label_annotations[data_type]) < data_type_count[data_type]:
    #                 _no_label_annotations[data_type].append(annotation)
    #             else:
    #                 _no_label_annotations['extra'].append(annotation)
    #         else:
    #             _no_label_annotations['extra'].append(annotation)
    #
    #     # we now add the NO_LABEL annotations with the desired data_type to our
    #     # "best" NO_LABELS
    #     for data_type, _ in data_type_count.most_common():
    #         best_no_label_annotations.extend(_no_label_annotations[data_type])
    #
    #     random.shuffle(best_no_label_annotations)
    #
    #     if len(best_no_label_annotations) >= n_no_labels:
    #         return best_no_label_annotations[:n_no_labels]
    #
    #     # if we still didn't have enough we append the 'extra' NO_LABEL annotations here
    #     best_no_label_annotations.extend(_no_label_annotations['extra'])
    #
    #     # we don't shuffle before we trim the array here so the 'extra' NO_LABEL annotations
    #     # are the ones being cut off at the end
    #     return best_no_label_annotations[:n_no_labels]

    # def predict_data_type(self, annotation):
    #     """Use in get_best_no_label."""
    #     if normalize_to_positive_float(annotation):
    #         return 'Positive Number'
    #     if normalize_to_float(annotation):
    #         return 'Number'
    #     if normalize_to_date(annotation):
    #         return 'Date'
    #     if normalize_to_bool(annotation):
    #         return 'True/False'
    #     if normalize_to_percentage(annotation):
    #         return 'Percentage'
    #     return 'Text'

    def train_valid_split(self):
        """Split documents randomly into valid and train data."""
        logger.info('Splitting into train and valid')

        # if we don't want to split into train/valid then set df_valid to empty df
        # if self.train_split_percentage == 1:
        self.df_valid = pd.DataFrame()
        # else:
        #     # else, first find labels which only appear once so can't be stratified
        #     single_labels = [
        #         lbl for (lbl, cnt) in self.df_train[self._LABEL_TARGET_COLUMN_NAME].value_counts().items() if cnt <= 1
        #     ]
        #     if single_labels:
        #         # if we find any, add to df_singles df
        #         logger.info(f'Following labels appear only once in df_train so are not in df_valid: {single_labels}')
        #         df_singles = self.df_train.groupby(self._LABEL_TARGET_COLUMN_NAME).filter(lambda x: len(x) == 1)
        #
        #     # drop labels that only appear once in df_train as they cannot be stratified
        #     self.df_train = self.df_train.groupby(self._LABEL_TARGET_COLUMN_NAME).filter(lambda x: len(x) > 1)
        #
        #     # do stratified split
        #     self.df_train, self.df_valid = train_test_split(
        #         self.df_train,
        #         train_size=self.train_split_percentage,
        #         stratify=self.df_train[self._LABEL_TARGET_COLUMN_NAME],
        #         random_state=self.label_random_state
        #     )
        #
        #     # if we found any single labels, add them back to df_train
        #     if single_labels:
        #         self.df_train = pd.concat([self.df_train, df_singles])

        if self.df_train.empty:
            raise Exception('Not enough data to train an AI model.')

        if self.df_train[self.label_feature_list].isnull().values.any():
            raise Exception('Sample with NaN within the training data found! Check code!')

        # if not self.df_valid.empty:
        #     if self.df_valid[self.label_feature_list].isnull().values.any():
        #         raise Exception('Sample with NaN within the validation data found! Check code!')

    def fit(self) -> RandomForestClassifier:
        """Given training data and the feature list this function returns the trained regression model."""
        logger.info('Start training of multiclass_clf.')

        self.clf = RandomForestClassifier(
            class_weight=self.label_class_weight,
            n_estimators=self.label_n_estimators,
            max_depth=self.label_max_depth,
            random_state=self.label_random_state
        )

        self.clf.fit(self.df_train[self.label_feature_list], self.df_train[self._LABEL_TARGET_COLUMN_NAME])
        return self.clf

    def evaluate(self):
        """Start evaluation."""
        if not self.df_test.empty:
            logger.error('The test set is empty. Skip evaluation for test data.')
        else:
            logger.info('Evaluating label classifier on the test data')
            self.df_test.loc[~self.df_test.is_correct, self._LABEL_TARGET_COLUMN_NAME] = self.no_label.name
            self._evaluate_multiclass_clf(self.df_test)

        # Run extraction on test documents.
        def _evaluate(document, model):
            return model.evaluate_extraction_model(document)

        pool = ProcessPool()
        data = pool.map(partial(_evaluate, model=self), self.category.documents())
        df_data = pd.concat(data)
        output_dir = self.category.project.model_folder
        file_path = os.path.join(output_dir, f'{get_timestamp()}.csv')
        df_data.to_csv(file_path)

        logger.info('Training Documents')
        Evaluation(df_data).label_evaluations(dataset_status=[2])

        logger.info('Test Documents')
        Evaluation(df_data).label_evaluations(dataset_status=[3])


    def _evaluate_multiclass_clf(self, df: pd.DataFrame):
        """
        Evaluate the label classifier on a given DataFrame.

        Evaluates by computing the accuracy, balanced accuracy and f1-score across all labels
        plus the f1-score, precision and recall across each label individually.
        """
        # copy the df as we do not want to modify it
        df = df.copy()

        # get probability of each class
        _results = pd.DataFrame(
            data=self.clf.predict_proba(X=df[self.label_feature_list]),
            columns=self.clf.classes_)

        # get predicted label index over all classes
        predicted_label_list = list(_results.idxmax(axis=1))
        # get predicted label probability over all classes
        # accuracy_list = list(_results.max(axis=1))
        #
        # # get another dataframe with only the probability over the classes that aren't NO_LABEL
        # _results_only_label = pd.DataFrame()
        # if 'NO_LABEL' in _results.columns:
        #     _results_only_label = _results.drop(['NO_LABEL'], axis=1)
        #
        # if _results_only_label.shape[1] > 0:
        #     # get predicted label index over all classes that are not NO_LABEL
        #     only_label_predicted_label_list = list(_results_only_label.idxmax(axis=1))
        #     # get predicted label probability over all classes that are not NO_LABEL
        #     only_label_accuracy_list = list(_results_only_label.max(axis=1))
        #
        #     # for each predicted label (over all classes)
        #     for index in range(len(predicted_label_list)):
        #         # if the highest probability to a non NO_LABEL class is >=0.2, we say it predicted that class instead
        #         # replace predicted label index and probability
        #         if only_label_accuracy_list[index] >= 0.2:
        #             predicted_label_list[index] = only_label_predicted_label_list[index]
        #             accuracy_list[index] = only_label_accuracy_list[index]
        # else:
        #     logger.info('\n[WARNING] _results_only_label is empty.\n')

        # add a column for predicted label index
        df.insert(
            loc=0,
            column='predicted_' + self._LABEL_TARGET_COLUMN_NAME,
            value=predicted_label_list
        )

        # # add a column for prediction probability (not actually accuracy)
        # df.insert(
        #     loc=0,
        #     column='confidence',
        #     value=accuracy_list
        # )

        # get and sort the importance of each feature
        feature_importances = self.clf.feature_importances_

        feature_importances_list = sorted(
            list(zip(self.label_feature_list, feature_importances)),
            key=lambda item: item[1],
            reverse=True
        )

        # computes the general metrics, i.e. across all labels
        y_true = df[self._LABEL_TARGET_COLUMN_NAME]
        y_pred = df['predicted_' + self._LABEL_TARGET_COLUMN_NAME]

        # gets accuracy, balanced accuracy and f1-score over all labels
        results_general = {
            'label_text': 'general/all annotations',
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1-score': f1_score(y_true, y_pred, average=self._GENERAL_RESULTS_AVERAGE_MODE),
        }

        # gets accuracy, balanced accuracy and f1-score over all labels (except for 'NO_LABEL'/'NO_LABEL')
        y_true_filtered = []
        y_pred_filtered = []
        for s_true, s_pred in zip(y_true, y_pred):
            if not (s_true == self.no_label.name and s_pred == self.no_label.name):
                y_true_filtered.append(s_true)
                y_pred_filtered.append(s_pred)

        results_general_filtered = {
            'label_text': 'all annotations except TP of NO_LABEL',
            'accuracy': accuracy_score(y_true_filtered, y_pred_filtered),
            'balanced accuracy': balanced_accuracy_score(y_true_filtered, y_pred_filtered),
            'f1-score': f1_score(y_true_filtered, y_pred_filtered, average=self._GENERAL_RESULTS_AVERAGE_MODE),
        }

        # compute all metrics again, but per label
        labels = list(set(df[self._LABEL_TARGET_COLUMN_NAME]))
        precision, recall, fscore, support = precision_recall_fscore_support(y_pred, y_true, labels=labels)

        # store results for each label
        results_labels_list = []

        for i, label in enumerate(labels):
            results = {
                'label_text': label,
                'accuracy': None,
                'balanced accuracy': None,
                'f1-score': fscore[i],
                'precision': precision[i],
                'recall': recall[i]
            }
            results_labels_list.append(results)

        # sort results for each label in descending order by their f1-score
        results_labels_list_sorted = sorted(results_labels_list, key=lambda k: k['f1-score'], reverse=True)

        # combine general results and label specific results into one dict
        self.results_summary = {
            'general': results_general,
            'general_filtered': results_general_filtered,
            'label-specific': results_labels_list_sorted
        }

        # get the probability_distribution
        prob_dict = self._get_probability_distribution(df)
        prob_list = [(k, v) for k, v in prob_dict.items()]
        prob_list.sort(key=lambda tup: tup[0])
        self.df_prob = pd.DataFrame(prob_list, columns=['Range of predicted Accuracy', 'Real Accuracy in this range'])

        # log results and feature importance and probability distribution as tables
        logger.info('\n' + tabulate(
            pd.DataFrame([results_general, results_general_filtered] + results_labels_list_sorted),
            floatfmt=".1%", headers="keys", tablefmt="pipe") + '\n')

        logger.info('\n' + tabulate(
            pd.DataFrame(feature_importances_list, columns=['feature_name', 'feature_importance']),
            floatfmt=".4%", headers="keys", tablefmt="pipe") + '\n')

        logger.info('\n' + tabulate(self.df_prob, floatfmt=".2%", headers="keys", tablefmt="pipe") + '\n')

        return self.results_summary

    def _get_probability_distribution(self, df):
        """Calculate the probability distribution according to the range of confidence."""
        # group by accuracy
        step_size = self._PROBABILITY_DISTRIBUTION_STEP
        step_list = np.arange(self._PROBABILITY_DISTRIBUTION_START, 1 + step_size, step_size)
        df_dict = {}
        for index, step in enumerate(step_list):
            if index + 1 < len(step_list):
                lower_bound = round(step, 2)
                upper_bound = round(step_list[index + 1], 2)
                df_range = df[(lower_bound < df['confidence']) & (df['confidence'] <= upper_bound)]
                df_range_acc = accuracy_score(
                    df_range[self._LABEL_TARGET_COLUMN_NAME], df_range['predicted_' + self._LABEL_TARGET_COLUMN_NAME]
                )
                df_dict[str(lower_bound) + '-' + str(upper_bound)] = df_range_acc

        return df_dict

    # not used anymore
    # def _filter_annotations_for_duplicates(self, doc_annotations_list: List['Annotation']):
    #     """
    #     Filter the annotations for duplicates.
    #
    #     A duplicate is characterized by having the same start_offset, end_offset and label_name.
    #     Duplicates have to be filtered as there should be only one logical truth per specific
    #     text_offset and label.
    #     """
    #     annotations_filtered = []
    #     res = collections.defaultdict(list)
    #
    #     number_of_duplicates = 0
    #
    #     for annotation in doc_annotations_list:
    #         key ='_'.join(f'{x.start_offset}:{x.end_offset}' for x in annotation._spans)
    #         res[key].append(annotation)
    #
    #     annotations_bundled = list(res.values())
    #     for annotation_cluster in annotations_bundled:
    #         if len(annotation_cluster) > 1:
    #             number_of_duplicates += 1
    #             found = False
    #             for annotation in annotation_cluster:
    #                 if annotation.label.id_:
    #                     found = True
    #                     annotations_filtered.append(annotation)
    #
    #             if found is False:
    #                 annotations_filtered.append(annotation_cluster[0])
    #
    #         else:
    #             annotations_filtered.append(annotation_cluster[0])
    #
    #     if number_of_duplicates:
    #         message = f'{number_of_duplicates} duplicated annotations found.'
    #         logger.error(message)
    #         raise Exception(message)
    #     return annotations_filtered

    # Label Set CLF starts here
    def fit_label_set_clf(self) -> Tuple[Optional[object], Optional[List['str']]]:
        """
        Fit classifier to predict start lines of AnnotationSets.

        :return:
        """
        # Only train label_set clf is there are non default label_sets
        # if len([x for x in self.label_sets if x.has_multiple_annotation_sets]) == 0:
        if len([x for x in self.label_sets if (not x.is_default) and (x.name is not None)]) == 0:
            logger.info('Label_set_clf is not fitted as there is no Label Set other than the default one.')
            # logger.info('Label_set_clf is not fitted as there is no Label Set with multiple Annotation Sets.')
            self.label_set_clf = None
            return None

        self.label_set_clf = RandomForestClassifier(
            n_estimators=self.label_set_n_estimators,
            max_depth=self.label_set_max_depth
        )

        logger.info('Start fitting process of label_set_clf.')
        # ignores the annotation_set count as it actually worsens results
        self.label_set_feature_list = [label.name for label in self.labels if label.id_]

        # df_train_label_set_list = []
        df_train_ground_truth_list = []

        for document_id, df_doc in self.df_train.groupby('document_id'):
            df_doc = df_doc.reset_index()
            document = self.category.project.get_document_by_id(document_id)
            # Train classifier only on documents with a matching document label_set.
            df_1 = self.convert_label_features_to_label_set_features(df_doc, document)
            # TODO: We already have the span features in the df_doc but we get them again from the document.
            #  What if the document is not available (virtual_doc)?
            df_2 = self.build_document_label_set_feature(document)
            # df_train_label_set_list.append(df_1)
            df_train_ground_truth_list.append(df_2)
            # try:
            #     assert sorted(list(df_1) + ['document', 'line', 'text', 'y']) == sorted(list(df_2))
            # except:
            #     print('.')

        # df_train_expanded_features_list = [
        #     self.generate_relative_line_features(pd.DataFrame(df, columns=self.label_set_feature_list))
        #     for df in df_train_label_set_list]
        # df_valid_expanded_features_list = [
        #     self.generate_relative_line_features(pd.DataFrame(df, columns=self.label_set_feature_list))
        #     for df in df_valid_label_set_list]

        df_train_ground_truth = pd.DataFrame(
            pd.concat(df_train_ground_truth_list),
            columns=self.label_set_feature_list + [self._LABEL_SET_TARGET_COLUMN_NAME]
        )

        # self.label_set_expanded_feature_list = list(df_train_expanded_features_list[0].columns)
        #
        # df_train_expanded_features = pd.DataFrame(pd.concat(df_train_expanded_features_list),
        #                                           columns=self.label_set_expanded_feature_list)
        # if len(df_valid_expanded_features_list) > 0:
        #     df_valid_expanded_features = pd.DataFrame(pd.concat(df_valid_expanded_features_list),
        #                                               columns=self.label_set_expanded_feature_list)

        y_train = np.array(df_train_ground_truth[self._LABEL_SET_TARGET_COLUMN_NAME]).astype('str')
        x_train = df_train_ground_truth[self.label_set_feature_list]

        # fillna(0) is used here as not every label is found in every document at least once
        x_train.fillna(self._DEFAULT_VALUE_MISSING_LABEL, inplace=True)

        # No features available
        if x_train.empty:
            logger.error('No features available to train Label Set classifier, '
                         'probably because there are no annotations.')
            return None, None

        self.label_set_clf.fit(x_train, y_train)

        # Test with validation dataset.
        if not self.df_valid.empty:
            # df_valid_label_set_list = []
            df_valid_ground_truth_list = []
            for document_id, df_doc in self.df_valid.groupby('document_id'):
                df_doc = df_doc.reset_index()
                document = self.category.project.get_document_by_id(document_id)
                df_1 = self.convert_label_features_to_label_set_features(df_doc, document.text)
                df_2 = self.build_document_label_set_feature(document)
                # df_valid_label_set_list.append(df_1)
                df_valid_ground_truth_list.append(df_2)

            df_valid_ground_truth = pd.DataFrame(
                pd.concat(df_valid_ground_truth_list),
                columns=self.label_set_feature_list + [self._LABEL_SET_TARGET_COLUMN_NAME]
            )

            y_valid = np.array(df_valid_ground_truth[self._LABEL_SET_TARGET_COLUMN_NAME]).astype('str')
            x_valid = df_valid_ground_truth[self.label_set_feature_list]
            x_valid.fillna(self._DEFAULT_VALUE_MISSING_LABEL, inplace=True)

            y_pred = self.label_set_clf.predict(x_valid)
            # evaluate the clf
            self.evaluate_label_set_clf(y_valid, y_pred, self.label_set_clf.classes_)

        return self.label_set_clf, self.label_set_feature_list

    # Seems hacky dont use for the moment.
    # def generate_relative_line_features(self, df_features: pd.DataFrame) -> pd.DataFrame:
    #     """Add the features of the n_nearest previous and next lines."""
    #     n_nearest = self.n_nearest_label_set
    #     if n_nearest == 0:
    #         return df_features
    #
    #     min_row = 0
    #     max_row = len(df_features.index) - 1
    #
    #     df_features_new_list = []
    #
    #     for index, row in df_features.iterrows():
    #         row_dict = row.to_dict()
    #
    #         # get a relevant lines and add them to the dict_list
    #         for i in range(n_nearest):
    #             if index + (i + 1) <= max_row:
    #                 d_next = df_features.iloc[index + (i + 1)].to_dict()
    #             else:
    #                 d_next = row.to_dict()
    #                 d_next = {k: 0 for k, v in d_next.items()}
    #             d_next = {f'next_line_{i + 1}_{k}': v for k, v in d_next.items()}
    #
    #             if index - (i + 1) >= min_row:
    #                 d_prev = df_features.iloc[index - (i + 1)].to_dict()
    #             else:
    #                 d_prev = row.to_dict()
    #                 d_prev = {k: 0 for k, v in d_prev.items()}
    #             d_prev = {f'prev_line_{i + 1}_{k}': v for k, v in d_prev.items()}
    #             # merge the line into the row dict
    #             row_dict = {**row_dict, **d_next, **d_prev}
    #
    #         df_features_new_list.append(row_dict)
    #
    #     return pd.DataFrame(df_features_new_list)

    def convert_label_features_to_label_set_features(self, feature_df_label: pd.DataFrame, document: Document) \
            -> pd.DataFrame:
        """
        Convert the feature_df for the label_clf to a feature_df for the label_set_clf.

        The input is the Feature-Dataframe and text for one document.
        """
        # reset indices to avoid bugs with stupid NaN's as label_text
        # feature_df_label.reset_index(drop=True, inplace=True)

        # TODO: why are we predicting it we are just converting features?
        # predict and transform the DataFrame to be compatible with the other functions
        results = pd.DataFrame(
            data=self.clf.predict_proba(X=feature_df_label[self.label_feature_list]),
            columns=self.clf.classes_
        )

        # Remove no_label predictions
        if self.no_label.name in results.columns:
            results = results.drop([self.no_label.name], axis=1)

        # Store most likely prediction and its accuracy in separated columns
        feature_df_label[self._LABEL_TARGET_COLUMN_NAME] = results.idxmax(axis=1)
        feature_df_label['confidence'] = results.max(axis=1)

        # convert the transformed df to the new label_set features
        df_ = self.build_document_label_set_feature_X(number_of_lines=document.number_of_lines, df=feature_df_label)

        # convert the transformed df to the new label_set features
        df = df_.filter(self.label_set_feature_list, axis=1)
        return df

    def evaluate_label_set_clf(self, y_true, y_pred, classes):
        """
        Evaluate a label_set clf by comparing the ground truth to the predictions.

        Classes are the different classes of the label_set clf (the different annotation_sets).
        """
        logger.info('Evaluate label_set classifier on the validation data.')

        try:
            matrix = pd.DataFrame(
                confusion_matrix(y_true=y_true, y_pred=y_pred, labels=classes),
                columns=classes,
                index=['y_true_' + x for x in classes]
            )
            logger.info('\n' + tabulate(matrix, headers=classes))
        except ValueError:
            pass

        logger.info(f'precision: {precision_score(y_true, y_pred, average=self._LABEL_SET_RESULTS_AVERAGE_MODE)}')
        logger.info(f'recall: {recall_score(y_true, y_pred, average=self._LABEL_SET_RESULTS_AVERAGE_MODE)}')

    def build_document_label_set_feature(self, document) -> pd.DataFrame():
        """Build document feature for label_set classifier given ground truth."""
        # df = pd.DataFrame()
        # char_count = 0

        document_annotations = [
            annotation for annotation_set in document.annotation_sets() for annotation in annotation_set.annotations
        ]
        spans = [span for annotation in document_annotations for span in annotation.spans]

        df_ = pd.DataFrame([x.eval_dict() for x in spans])
        df_ = df_[~df_['label_id'].isnull()]

        # todo add to eval_dict.
        df_[self._LABEL_TARGET_COLUMN_NAME] = df_['label_id'].apply(
            lambda x: x if pd.isna(x) else self.category.project.get_label_by_id(x).name
        )

        df = self.build_document_label_set_feature_X(df=df_, number_of_lines=document.number_of_lines)

        # TODO raise exception when empty start_offset
        start_line_indexes = dict(
            (x.start_line_index, x.label_set.name)
            for x in document.annotation_sets()
            if x.start_line_index and x.label_set.id_
        )

        df[self._LABEL_SET_TARGET_COLUMN_NAME] = self._LABEL_SET_TARGET_DEFAULT_VALUE
        df.loc[start_line_indexes.keys(), self._LABEL_SET_TARGET_COLUMN_NAME] = list(start_line_indexes.values())
        return df

        # # todo raise error for overlap of annotation sets.
        # line_annotation_dict = {}
        # for i, line in enumerate(document.text.replace('\f', '\n').split('\n')):
        #     new_char_count = char_count + len(line)
        #     line_annotations = []
        #     for annotation in document_annotations:
        #         for span in annotation._spans:
        #             if char_count <= span.start_offset < new_char_count:
        #                 line_annotations.append(annotation)
        #     line_annotation_dict[i] = line_annotations
        #     char_count = new_char_count + 1
        #
        # # Loop over lines
        # char_count = 0
        # for i, line in enumerate(document.text.replace('\f', '\n').split('\n')):
        #     matched_annotation_set = None
        #     new_char_count = char_count + len(line)
        #     assert line == document.text[char_count: new_char_count]
        #     # TODO: Currently we can't handle
        #     for annotation_set in document.annotation_sets():
        #         # TODO: start_offset on Annotation Level is legacy #8820
        #         if annotation_set.start_offset and char_count <= annotation_set.start_offset < new_char_count:
        #             matched_annotation_set = annotation_set
        #             break
        #
        #     line_annotations = line_annotation_dict[i]
        #     annotations_dict = dict((x.label.name, True) for x in line_annotations)
        #     counter_dict = dict(Counter(x.label_set.name for x in line_annotations))
        #     y = matched_annotation_set.label_set.name if matched_annotation_set else 'No'
        #     df = df.append({'line': i, 'y': y, 'document': document.id_, **annotations_dict, **counter_dict},
        #                    ignore_index=True)
        #     char_count = new_char_count + 1
        # df['text'] = document.text.replace('\f', '\n').split('\n')
        # return df.fillna(0)

    def build_document_label_set_feature_X(self, number_of_lines: int, df: pd.DataFrame) -> pd.DataFrame():
        """
        Calculate features for a document given the extraction results.

        :param number_of_lines: number of lines of the document
        :param df:
        :return:
        """
        global_df = pd.DataFrame()

        # Using OptimalThreshold is a bad idea as it might defer between training (actual threshold from the label)
        # and runtime (default threshold.

        # TODO can this be replaced by something like.
        # df.groupby(['line_index', 'label_text']).size().reset_index(name='counts')

        # We group the results per line_index and we count the number of spans for each Label and Label Set of the
        # model per line.
        # We build a df where the rows correspond to a line_index and the columns to the Labels and Label Sets in the
        # model. The values are the number of spans. If a Label is shared between Label Sets and the Label Sets
        # belong to the same category, the number of spans will be added in all.
        for i, line_df in df.groupby('line_index'):
            counter_dict = collections.defaultdict(lambda: 0)
            annotations_dict = collections.defaultdict(lambda: 0)
            for label_text in list(line_df[self._LABEL_TARGET_COLUMN_NAME]):
                label = next(x for x in self.labels if x.name == label_text)
                annotations_dict[label.name] += 1
                for label_set in label.label_sets:
                    # TODO: what happens if a label does not belong to any label set in the category?
                    if label_set in self.label_sets:
                        counter_dict[label_set.name] += 1

            new_row = pd.Series(data={**annotations_dict, **counter_dict}, name=i)
            global_df = global_df.append(new_row, ignore_index=False)

        if not set(global_df.columns.to_list()).issubset(
                [x.name for x in self.labels] + [s.name for s in self.label_sets]
        ):
            raise ValueError(
                f'There are columns in the features for the Label Set that do not correspond to a Label or '
                f'Label Set from the Category: {set(global_df.columns.to_list())}')

        # We add a column for document_id and line_index. The line_index will correspond to the index of the global_df.
        df = global_df.reindex(
            columns=['document_id', 'line_index'] + self.label_set_feature_list,
            fill_value=self._LABEL_SET_FEATURE_DEFAULT_VALUE
        ).reindex(
            range(number_of_lines), fill_value=self._LABEL_SET_FEATURE_DEFAULT_VALUE
        )

        return df.fillna(self._LABEL_SET_FEATURE_DEFAULT_VALUE)

    def extract_label_set_with_clf(self, document, df, res_dict):
        """Run label_set classifier to calculate annotation_sets."""
        logger.info('Extract annotation_sets.')

        if self.label_set_clf:
            df_ = df[df['confidence'] >= self.label_set_confidence_threshold]  # df['OptimalThreshold']]
            feature_df = self.build_document_label_set_feature_X(
                number_of_lines=document.number_of_lines, df=df_
            ).filter(self.label_set_feature_list, axis=1)
            feature_df = feature_df.reindex(
                columns=self.label_set_feature_list
            ).fillna(self._LABEL_SET_FEATURE_DEFAULT_VALUE)
            # feature_df = self.generate_relative_line_features(feature_df)

            res_series = self.label_set_clf.predict(feature_df)
            res_label_sets = pd.DataFrame(res_series)
        else:
            return res_dict

        return self.extract_from_label_set_output(res_dict, res_label_sets)

    def extract_from_label_set_output(self, res_dict: dict, res_label_sets: pd.DataFrame, choose_top: bool = False) \
            -> dict:
        logger.info('Building new res dict')
        new_res_dict = {}
        # text_replaced = document.text.replace('\f', '\n')

        if res_label_sets.empty:
            raise ValueError('Label Set Classifier result is empty and it should have the default value "No".')
        if not res_dict:
            return {}
        working_res_dict = res_dict.copy()

        get_labelset_by_name = {labelset.name: labelset for labelset in self.label_sets}
        candidate_ann_sets = {labelset.name: [] for labelset in self.label_sets if labelset.name is not None}
        detected_lines = res_label_sets.loc[res_label_sets[0] != 'No']
        for i, this_line_num in enumerate(detected_lines.index):
            # we try to find the labels that match that annotation_set
            this_annset_name = detected_lines.loc[this_line_num][0]
            next_line_num = None
            try:
                next_line_num = detected_lines.index[i + 1]
            except IndexError:
                next_line_num = len(res_label_sets)
            start_line_this_annset = this_line_num + 1
            end_line_this_annset = next_line_num

            # possible_anns_for_this_annset = {label.name:None for label in get_labelset_by_name[this_annset_name].labels}
            possible_anns_for_this_annset = {}
            # possible_anns_for_this_annset["__start__"] = this_line_num
            # possible_anns_for_this_annset["__end__"] = end_line_this_annset
            # we get the label df that is contained within the annotation_set
            for label in get_labelset_by_name[this_annset_name].labels:
                if label.name in working_res_dict.keys():
                    label_df = working_res_dict[label.name]
                    label_df = label_df[
                        (start_line_this_annset <= label_df['line_index']) &
                        (label_df['line_index'] <= end_line_this_annset)
                        ]
                    if label_df.empty:
                        continue
                    possible_anns_for_this_annset[label.name] = label_df
                    # Remove from input dict
                    working_res_dict[label.name] = working_res_dict[label.name].drop(label_df.index)
            candidate_ann_sets[this_annset_name].append(possible_anns_for_this_annset)

        def avg_conf(current_ann_set: dict):
            if not current_ann_set:
                return 0
            total_conf_across_labels = 0
            for label_name_, label_df_ in current_ann_set.items():
                label_avg = sum(label_df_["confidence"]) / len(label_df_["confidence"])
                total_conf_across_labels += label_avg
            return total_conf_across_labels / len(current_ann_set)

        def highest_ann_conf(label_df_: pd.DataFrame):
            highest_line_ = label_df_.query('confidence == confidence.max()')
            return highest_line_, highest_line_.index

        def highest_avg_conf(detected: List):
            ann_set_confidences = [avg_conf(ann_set_) for ann_set_ in detected]
            max_avg_conf = max(ann_set_confidences)
            max_index = ann_set_confidences.index(max_avg_conf)
            max_ann_set = detected[max_index]
            del detected[max_index]
            return max_ann_set, detected

        def merge_default_ann_sets(detected: List):  # detected_annotation_sets
            merged = {}
            for ann_set_ in detected:
                for label_name_, label_df_ in ann_set_.items():
                    if label_name_ not in merged:
                        merged[label_name_] = pd.DataFrame(columns=label_df_.columns)
                    merged[label_name_] = merged[label_name_].append(label_df_.copy())
            return merged

        # Add Extraction from LabelSets with multiple annotation_sets (as list).
        for label_set in [x for x in self.label_sets if not x.is_default]:
            if (label_set.name is None) and (None not in candidate_ann_sets):
                continue
            if label_set.has_multiple_annotation_sets:
                new_res_dict[label_set.name] = []
                detected_annotation_sets = candidate_ann_sets[label_set.name]
                for annotation_set in detected_annotation_sets:
                    if annotation_set:
                        new_res_dict[label_set.name].append(annotation_set)
        # Add Extraction from LabelSets with single annotation_set (as dict).
        for label_set in [x for x in self.label_sets if not x.is_default]:
            if (label_set.name is None) and (None not in candidate_ann_sets):
                continue
            if not label_set.has_multiple_annotation_sets:
                detected_annotation_sets = candidate_ann_sets[label_set.name]
                if detected_annotation_sets:
                    highest_ann_set, others = highest_avg_conf(detected_annotation_sets)
                    if highest_ann_set:
                        new_res_dict[label_set.name] = highest_ann_set
                    # release the other ann sets to "No"
                    for ann_set in others:
                        for label_name in ann_set:
                            if label_name in working_res_dict:
                                label_df = ann_set[label_name]
                                working_res_dict[label_name] = working_res_dict[label_name].append(label_df)
        # Add Extraction from the default LabelSet.
        for label_set in [x for x in self.label_sets if x.is_default]:
            detected_annotation_sets = candidate_ann_sets[label_set.name]
            if detected_annotation_sets:
                merged_default_ann_set = merge_default_ann_sets(detected_annotation_sets)
                for label_name in merged_default_ann_set:
                    new_res_dict[label_name] = merged_default_ann_set[label_name]
        # Add remaining extractions
        for label_set in [x for x in self.label_sets if not x.is_default]:
            if label_set.has_multiple_annotation_sets:
                _dict = {}
                for label in label_set.labels:
                    if label.name in working_res_dict.keys():
                        if not working_res_dict[label.name].empty:
                            _dict[label.name] = working_res_dict[label.name].copy()
                            working_res_dict[label.name] = working_res_dict[label.name].drop(
                                working_res_dict[label.name].index)
                if _dict:
                    new_res_dict[label_set.name].append(_dict)
        for label_set in [x for x in self.label_sets if not x.is_default]:
            if not label_set.has_multiple_annotation_sets:
                _dict = {}
                for label in label_set.labels:
                    if label.name in working_res_dict.keys():
                        # collect all or just top predictions
                        if choose_top:
                            highest_line, highest_index = highest_ann_conf(working_res_dict[label.name])
                            if not highest_line.empty:
                                _dict[label.name] = highest_line
                                working_res_dict[label.name] = working_res_dict[label.name].drop(highest_index)
                        else:
                            if not working_res_dict[label.name].empty:
                                _dict[label.name] = working_res_dict[label.name].copy()
                                working_res_dict[label.name] = working_res_dict[label.name].drop(
                                    working_res_dict[label.name].index)
                if _dict:
                    new_res_dict[label_set.name] = _dict
        # Finally, add remaining extractions to default annotation_set (if they are allowed to be there).
        for label_set in [x for x in self.label_sets if x.is_default]:
            for label in label_set.labels:
                if label.name in working_res_dict.keys():
                    # collect all or just top predictions
                    if choose_top:
                        highest_line, highest_index = highest_ann_conf(working_res_dict[label.name])
                        if not highest_line.empty:
                            new_res_dict[label.name] = highest_line
                            working_res_dict[label.name] = working_res_dict[label.name].drop(highest_index)
                    else:
                        if label.name not in new_res_dict:
                            new_res_dict[label.name] = pd.DataFrame(columns=working_res_dict[label.name].columns)
                        new_res_dict[label.name] = new_res_dict[label.name].append(working_res_dict[label.name].copy())
                        working_res_dict[label.name] = working_res_dict[label.name].drop(
                            working_res_dict[label.name].index)

        return new_res_dict
