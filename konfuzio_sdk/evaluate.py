"""Calculate the accuracy on any level in a  Document."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.extmath import weighted_mode

from konfuzio_sdk.data import Category, Document
from konfuzio_sdk.utils import memory_size_of, sdk_isinstance

logger = logging.getLogger(__name__)

RELEVANT_FOR_EVALUATION = [
    'is_matched',  # needed to group spans in Annotations
    'id_local',  # needed to group spans in Annotations
    'id_',  # even we won't care of the id_, as the ID is defined by the start and end span
    # "confidence", we don't care about the confidence of doc_a
    'start_offset',  # only relevant for the merge but allows to track multiple sequences per annotation
    'end_offset',  # only relevant for the merge but allows to track multiple sequences per annotation
    'is_correct',  # we care if it is correct, humans create Annotations without confidence
    'label_id',
    'label_threshold',
    'above_predicted_threshold',
    'revised',  # we need it to filter feedback required Annotations
    'annotation_set_id',
    'label_set_id',
    'document_id',
    'document_id_local',
    'category_id',  # Identify the Category to be able to run an evaluation across categories
    # "id__predicted", we don't care of the id_ see "id_"
    'id_local_predicted',
    'confidence_predicted',  # we care about the confidence of the prediction
    'start_offset_predicted',
    'end_offset_predicted',
    # "is_correct_predicted", # it's a prediction so we don't know if it is correct
    'label_id_predicted',
    'label_has_multiple_top_candidates_predicted',
    'label_threshold_predicted',  # we keep a flexibility to be able to predict the threshold
    # "revised_predicted",  # it's a prediction so we ignore if it is revised
    'annotation_set_id_predicted',
    'label_set_id_predicted',
    'document_id_predicted',
    'document_id_local_predicted',
    'is_correct_label',
    'is_correct_label_set',
    'is_correct_annotation_set_id',
    'is_correct_id_',
    'duplicated',
    'duplicated_predicted',
    'tmp_id_',  # a temporary ID used for enumerating the predicted annotations solely
    'disambiguated_id',  # an ID for multi-span annotations
]

logger = logging.getLogger(__name__)


def grouped(group, target: str):
    """Define which of the correct element in the predicted group defines the "correct" group id_."""
    verbose_validation_column_name = f'defined_to_be_correct_{target}'
    # all rows where is_correct is nan relate to an element which has no correct element partner
    eligible_to_vote = group['above_predicted_threshold'].fillna(False) & group['is_matched'].fillna(False)
    if not eligible_to_vote.any():  # no Spans provide confidence above Threshold of Label
        if not group['confidence_predicted'].any() or not group[target].any():
            group[verbose_validation_column_name] = group[target].mode(dropna=False)[0]
        else:
            group[verbose_validation_column_name] = int(weighted_mode(group[target], group['confidence_predicted'])[0])
    elif not group.loc[eligible_to_vote][target].any():  # no Spans should be predicted
        group[verbose_validation_column_name] = None
    else:  # get the most frequent annotation_set_id from the high confidence Spans in this group
        if not group.loc[eligible_to_vote]['confidence_predicted'].any():  # CLF does not provide a confidence score
            # get the most frequent annotation_set_id from the *correct* Annotations in this group
            group[verbose_validation_column_name] = group.loc[eligible_to_vote][target].mode(dropna=False)[0]
        else:
            group[verbose_validation_column_name] = int(
                weighted_mode(group.loc[eligible_to_vote][target], group.loc[eligible_to_vote]['confidence_predicted'])[
                    0
                ]
            )
    validation_column_name = f'is_correct_{target}'
    group[validation_column_name] = group[target] == group[verbose_validation_column_name]
    return group


def prioritize_rows(group):
    """
    Apply a filter when a Label should only appear once per AnnotationSet but has been predicted multiple times.

    After we have calculated the TPs, FPs, FNs for the Document, we filter out the case where a Label should
    only appear once per AnnotationSet but has been predicted multiple times. In this case, if any of the
    predictions is a TP then we keep one and discard FPs/FNs. If no TPs, if any of the predictions is a FP
    then we keep one and discard the FNs. If no FPs, then we keep a FN. The prediction we keep is always the
    first in terms of start_offset.
    """
    group = group[~(group['label_has_multiple_top_candidates_predicted'].astype(bool))]
    if group.empty:
        return group

    first_true_positive = group[group['true_positive']].head(1)
    first_false_positive = group[group['false_positive']].head(1)
    first_false_negative = group[group['false_negative']].head(1)
    if not first_true_positive.empty:
        return first_true_positive
    elif not first_false_positive.empty:
        return first_false_positive
    else:
        return first_false_negative


def compare(
    doc_a,
    doc_b,
    only_use_correct=False,
    use_view_annotations=False,
    ignore_below_threshold=False,
    strict=True,
    id_counter: int = 1,
    custom_threshold=None,
) -> pd.DataFrame:
    """Compare the Annotations of two potentially empty Documents wrt. to **all** Annotations.

    :param doc_a: Document which is assumed to be correct
    :param doc_b: Document which needs to be evaluated
    :param only_use_correct: Unrevised feedback in doc_a is assumed to be correct.
    :param use_view_annotations: Will filter for top confidence annotations. Only available when strict=True.
     When use_view_annotations=True, it will compare only the highest confidence extractions to the ground truth
     Annotations. When False (default), it compares all extractions to the ground truth Annotations. This setting is
     ignored when strict=False, as the Non-Strict Evaluation needs to compare all extractions.
     For more details see https://help.konfuzio.com/modules/extractions/index.html#evaluation
    :param ignore_below_threshold: Ignore Annotations below detection threshold of the Label (only affects TNs)
    :param strict: Evaluate on a Character exact level without any postprocessing, an amount Span "5,55 " will not be
     exact with "5,55"
    :raises ValueError: When the Category differs.
    :return: Evaluation DataFrame
    """
    df_a = pd.DataFrame(doc_a.eval_dict(use_correct=only_use_correct))
    df_a_ids = df_a[['id_']]
    duplicated_ids = df_a_ids['id_'].duplicated(keep=False)
    df_a_ids['disambiguated_id'] = df_a_ids['id_'].astype(str)
    df_a_ids.loc[duplicated_ids, 'disambiguated_id'] += '_' + (df_a_ids.groupby('id_').cumcount() + 1).astype(str)
    df_a['disambiguated_id'] = df_a_ids['disambiguated_id']
    df_b = pd.DataFrame(
        doc_b.eval_dict(
            use_view_annotations=strict and use_view_annotations,  # view_annotations only available for strict=True
            use_correct=False,
            ignore_below_threshold=ignore_below_threshold,
        ),
    )
    df_b['tmp_id_'] = list(range(id_counter, id_counter + len(df_b)))

    if doc_a.category != doc_b.category:
        raise ValueError(f'Categories of {doc_a} with {doc_a.category} and {doc_b} with {doc_a.category} do not match.')
    if strict:  # many to many inner join to keep all Spans of both Documents
        spans = pd.merge(df_a, df_b, how='outer', on=['start_offset', 'end_offset'], suffixes=('', '_predicted'))
        # add criteria to evaluate Spans
        spans['is_matched'] = spans['id_local'].notna()  # start and end offset are identical
        spans['start_offset_predicted'] = spans['start_offset']  # start and end offset are identical
        spans['end_offset_predicted'] = spans['end_offset']  # start and end offset are identical

        if custom_threshold:
            spans['above_predicted_threshold'] = spans['confidence_predicted'] >= custom_threshold
        else:
            spans['above_predicted_threshold'] = spans['confidence_predicted'] >= spans['label_threshold_predicted']

        spans['is_correct_label'] = spans['label_id'] == spans['label_id_predicted']
        spans['is_correct_label_set'] = spans['label_set_id'] == spans['label_set_id_predicted']
        spans['duplicated'] = False
        spans['duplicated_predicted'] = False

        # add check to evaluate multiline Annotations
        spans = spans.groupby('id_local', dropna=False).apply(lambda group: grouped(group, 'id_'))
        # add check to evaluate Annotation Sets
        spans = spans.groupby('annotation_set_id_predicted', dropna=False).apply(
            lambda group: grouped(group, 'annotation_set_id')
        )
    else:
        # allows  start_offset_predicted <= end_offset and end_offset_predicted >= start_offset
        spans = pd.merge(df_a, df_b, how='outer', on=['label_id', 'label_set_id'], suffixes=('', '_predicted'))
        # add criteria to evaluate Spans
        spans['is_matched'] = (spans['start_offset_predicted'] <= spans['end_offset']) & (
            spans['end_offset_predicted'] >= spans['start_offset']
        )
        if custom_threshold:
            spans['above_predicted_threshold'] = spans['confidence_predicted'] >= custom_threshold
        else:
            spans['above_predicted_threshold'] = spans['confidence_predicted'] >= spans['label_threshold_predicted']
        spans['is_correct_label'] = True
        spans['is_correct_label_set'] = True
        spans['label_id_predicted'] = spans['label_id']
        spans['label_set_id_predicted'] = spans['label_set_id']

        spans = spans.sort_values(by='is_matched', ascending=False)
        spans['duplicated'] = spans.duplicated(subset=['id_local'], keep='first')
        spans['duplicated_predicted'] = spans.duplicated(subset=['id_local_predicted'], keep='first')
        spans = spans.drop(spans[(spans['duplicated']) & (spans['duplicated_predicted'])].index)
        # add check to evaluate multiline Annotations
        spans = spans.groupby('id_local', dropna=False).apply(lambda group: grouped(group, 'id_'))
        # add check to evaluate Annotation Sets
        spans = spans.groupby('annotation_set_id_predicted', dropna=False).apply(
            lambda group: grouped(group, 'annotation_set_id')
        )
    spans = spans[RELEVANT_FOR_EVALUATION]

    assert not spans.empty  # this function must be able to evaluate any two docs even without annotations

    spans['tokenizer_true_positive'] = (
        (spans['is_correct'])
        & (spans['is_matched'])
        & (spans['start_offset_predicted'] == spans['start_offset'])
        & (spans['end_offset_predicted'] == spans['end_offset'])
        & (spans['document_id_local_predicted'].notna())
    )

    spans['tokenizer_false_negative'] = (
        (spans['is_correct']) & (spans['is_matched']) & (spans['document_id_local_predicted'].isna())
    )

    spans['tokenizer_false_positive'] = (
        (~spans['tokenizer_false_negative'])
        & (~spans['tokenizer_true_positive'])
        & (spans['document_id_local_predicted'].notna())
        & (spans['end_offset'] != 0)  # ignore placeholder
    )

    spans['clf_true_positive'] = (
        (spans['is_correct'])
        & (spans['is_matched'])
        & (spans['document_id_local_predicted'].notna())
        & (spans['above_predicted_threshold'])
        & (spans['is_correct_label'])
    )

    spans['clf_false_negative'] = (
        (spans['is_correct'])
        & (spans['is_matched'])
        & (spans['document_id_local_predicted'].notna())
        & (~spans['above_predicted_threshold'])
        & (spans['is_correct_label'])
    )

    spans['clf_false_positive'] = (
        (spans['is_correct'])
        & (spans['is_matched'])
        & (spans['document_id_local_predicted'].notna())
        & (~spans['is_correct_label'])
    )

    # Evaluate which **spans** are TN, TP, FP and keep RELEVANT_FOR_MAPPING to allow grouping of confidence measures
    spans['true_positive'] = (
        (spans['is_matched'])
        & (spans['is_correct'])
        & (spans['above_predicted_threshold'])
        & (~spans['duplicated'])
        & (  # Everything is correct
            (spans['is_correct_label'])
            & (spans['is_correct_label_set'])
            & (spans['is_correct_annotation_set_id'])
            & (spans['is_correct_id_'])
        )
    )

    spans['false_negative'] = (
        (spans['is_correct'])
        & (~spans['duplicated'])
        & ((~spans['is_matched']) | (~spans['above_predicted_threshold']) | (spans['label_id_predicted'].isna()))
    )

    spans['false_positive'] = (
        (spans['above_predicted_threshold'])
        & (~spans['false_negative'])
        & (~spans['true_positive'])
        & (~spans['duplicated_predicted'])
        & (
            (~spans['is_correct_label'])
            | (~spans['is_correct_label_set'])
            | (~spans['is_correct_annotation_set_id'])
            | (~spans['is_correct_id_'])
            | (~spans['is_matched'])
        )
    )

    if not strict:
        # Apply the function prioritize_rows just to entries where the label is not set to "multiple"
        labels = doc_a.project.labels
        label_ids_multiple = [label.id_ for label in labels if label.has_multiple_top_candidates]
        label_ids_not_multiple = [label.id_ for label in labels if not label.has_multiple_top_candidates]
        spans_not_multiple = spans[spans['label_id'].isin(label_ids_not_multiple)]
        spans_not_multiple = spans_not_multiple.groupby(['annotation_set_id_predicted', 'label_id_predicted']).apply(
            prioritize_rows
        )
        spans_multiple = spans[spans['label_id'].isin(label_ids_multiple)]
        spans = pd.concat([spans_not_multiple, spans_multiple])
        spans = spans.sort_values(by='is_matched', ascending=False)

    spans = spans.replace({np.nan: None})
    # how many times annotations with this label occur in the ground truth data
    spans['frequency'] = spans.groupby('label_id')['label_id'].transform('size')
    spans['frequency'].fillna(0, inplace=True)
    spans['frequency'] = spans['frequency'].apply(lambda x: int(x))

    if not strict:
        # one Span must not be defined as TP or FP or FN more than once
        quality = (spans[['true_positive', 'false_positive', 'false_negative']].sum(axis=1) <= 1).all()
        assert quality
    return spans


class ExtractionConfusionMatrix:
    """Check how all predictions are mapped to the ground-truth Annotations."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the class.

        :param data: Raw evaluation data.
        """
        self.matrix = self.calculate(data=data)

    def calculate(self, data: pd.DataFrame):
        """
        Calculate the matrix.

        :param data: Raw evaluation data.
        """
        data = data.reset_index(drop=True)
        data['id_'] = data['id_'].fillna('no_match', inplace=True)
        data['tmp_id_'] = data['tmp_id_'].fillna('no_match')

        data['relation'] = data.apply(
            lambda x: 'TP'
            if x['true_positive']
            else ('FP' if x['false_positive'] else ('FN' if x['false_negative'] else 'TN')),
            axis=1,
        )

        matrix = pd.pivot(data, index='disambiguated_id', columns='tmp_id_', values=['relation'])
        matrix.fillna('TN', inplace=True)
        return matrix


class EvaluationCalculator:
    """Calculate precision, recall, f1, based on TP, FP, FN."""

    def __init__(self, tp: int = 0, fp: int = 0, fn: int = 0, tn: int = 0, zero_division='warn'):
        """
        Store evaluation information.

        :param tp: True Positives.
        :param fp: False Positives.
        :param fn: False Negatives.
        :param tn: True Negatives.
        :param zero_division: Defines how to handle situations when precision, recall or F1 measure calculations result
        in zero division.
        Possible values: 'warn' – log a warning and assign a calculated metric a value of 0.
        0 - assign a calculated metric a value of 0.
        'error' – raise a ZeroDivisionError.
        None – assign None to a calculated metric.
        """
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
        assert zero_division in ['warn', 'error', 0, None], (
            "The value of zero_division has to be 'warn', 'error', 0 " 'or None'
        )
        self.zero_division = zero_division

    @property
    def precision(self) -> Optional[float]:
        """
        Apply precision formula.

        :raises ZeroDivisionError: When TP and FP are 0 and zero_division is set to 'error'
        """
        if self.tp + self.fp != 0:
            precision = self.tp / (self.tp + self.fp)
        else:
            if self.zero_division == 'error':
                raise ZeroDivisionError('TP and FP are zero, impossible to calculate precision.')
            elif self.zero_division == 'warn':
                precision = 0
                logging.warning('TP and FP are zero, precision is set to 0.')
            else:
                precision = self.zero_division
        return precision

    @property
    def recall(self) -> Optional[float]:
        """
        Apply recall formula.

        :raises ZeroDivisionError: When TP and FN are 0 and zero_division is set to 'error'
        """
        if self.tp + self.fn != 0:
            recall = self.tp / (self.tp + self.fn)
        else:
            if self.zero_division == 'error':
                raise ZeroDivisionError('TP and FN are zero, recall is impossible to calculate.')
            elif self.zero_division == 'warn':
                recall = 0
                logging.warning('TP and FN are zero, recall is set to 0.')
            else:
                recall = self.zero_division
        return recall

    @property
    def f1(self) -> Optional[float]:
        """
        Apply F1-score formula.

        :raises ZeroDivisionError: When precision and recall are 0 and zero_division is set to 'error'
        """
        if self.tp + 0.5 * (self.fp + self.fn) != 0:
            f1 = self.tp / (self.tp + 0.5 * (self.fp + self.fn))
        else:
            if self.zero_division == 'error':
                raise ZeroDivisionError('Precision and recall are zero, F1 is impossible to calculate.')
            elif self.zero_division == 'warn':
                f1 = 0
                logging.warning('Precision and recall are zero, F1 score is set to 0.')
            else:
                f1 = self.zero_division
        return f1

    def metrics_logging(self):
        """Log metrics."""
        logger.info(f'true positives: {self.tp}')
        logger.info(f'false negatives: {self.fn}')
        logger.info(f'true negatives: {self.tn}')
        logger.info(f'false positives: {self.fp}')
        logger.info(f'precision: {self.precision}')
        logger.info(f'recall: {self.recall}')
        logger.info(f'F1: {self.f1}')


class ExtractionEvaluation:
    """Calculated accuracy measures by using the detailed comparison on Span Level."""

    def __init__(
        self,
        documents: List[Tuple[Document, Document]],
        strict: bool = True,
        use_view_annotations: bool = True,
        ignore_below_threshold: bool = True,
        zero_division='warn',
    ):
        """
        Relate to the two document instances.

        :param documents: A list of tuple Documents that should be compared.
        :param strict: Evaluate on a Character exact level without any postprocessing.
        :param zero_division: Defines how to handle situations when precision, recall or F1 measure calculations result
        in zero division.
        Possible values: 'warn' – log a warning and assign a calculated metric a value of 0.
        0 - assign a calculated metric a value of 0.
        'error' – raise a ZeroDivisionError.
        None – assign None to a calculated metric.
        :param use_view_annotations:
            Bool for whether to filter evaluated Document with view_annotations. Will filter out all overlapping Spans
            and below threshold Annotations. Should lead to faster evaluation.
        :param ignore_below_threshold:
            If true, will ignore all Annotations below the Label detection threshold. Only affects True Negatives.
            Leads to faster evaluation.
        """
        logger.info(f'Initializing Evaluation object with {len(documents)} documents. Evaluation mode {strict=}.')
        self.documents = documents
        self.strict = strict
        self.use_view_annotations = use_view_annotations
        self.ignore_below_threshold = ignore_below_threshold
        self.only_use_correct = True
        self.data = None
        self.zero_division = zero_division
        self.calculate()
        self.label_thresholds = {}
        self.calculate_thresholds()
        logger.info(f'Size of evaluation DataFrame: {memory_size_of(self.data)/1000} KB.')

    def calculate(self):
        """Calculate and update the data stored within this Evaluation."""
        evaluations = []  # start anew, the configuration of the Evaluation might have changed.
        id_counter = 1
        for ground_truth, predicted in self.documents:
            evaluation = compare(
                doc_a=ground_truth,
                doc_b=predicted,
                only_use_correct=self.only_use_correct,
                strict=self.strict,
                use_view_annotations=self.use_view_annotations,
                ignore_below_threshold=self.ignore_below_threshold,
                id_counter=id_counter,
            )
            evaluations.append(evaluation)
            id_counter += len(evaluation)

        self.data = pd.concat(evaluations)

    def calculate_thresholds(self):
        """
        Calculate optimal thresholds for each Label in the Document set that allow to achieve the highest value
        of F1 score, precision and recall.
        """
        evaluations_per_threshold = {}
        for threshold in np.arange(0.05, 1, 0.05):
            evaluations = []
            for ground_truth, predicted in self.documents:
                evaluation = compare(
                    doc_a=ground_truth,
                    doc_b=predicted,
                    only_use_correct=self.only_use_correct,
                    strict=self.strict,
                    use_view_annotations=self.use_view_annotations,
                    ignore_below_threshold=self.ignore_below_threshold,
                    custom_threshold=threshold,
                )
                evaluations.append(evaluation)
            evaluations_per_threshold[threshold] = pd.concat(evaluations)
        for label in self.documents[0][0].project.labels:
            self.label_thresholds[label.id_] = {}
            best_f1 = 0.0
            best_threshold_f1 = 0.0
            best_precision = 0.0
            best_threshold_precision = 0.0
            best_recall = 0.0
            best_threshold_recall = 0.0
            for threshold in evaluations_per_threshold:
                current_evaluation = evaluations_per_threshold[threshold]
                label_data = current_evaluation.query(f'label_id == {label.id_} | (label_id_predicted == {label.id_})')
                label_evaluation_calculator = EvaluationCalculator(
                    tp=label_data['true_positive'].sum(),
                    fp=label_data['false_positive'].sum(),
                    fn=label_data['false_negative'].sum(),
                    zero_division=self.zero_division,
                )
                label_f1 = label_evaluation_calculator.f1
                label_precision = label_evaluation_calculator.precision
                label_recall = label_evaluation_calculator.recall
                if label_f1 and label_f1 > best_f1:
                    best_f1 = label_f1
                    best_threshold_f1 = threshold
                if label_precision and label_precision > best_precision:
                    best_precision = label_precision
                    best_threshold_precision = threshold
                if label_recall and label_recall > best_recall:
                    best_recall = label_recall
                    best_threshold_recall = threshold
            label.optimized_thresholds['f1'] = {'score': best_f1, 'threshold': best_threshold_f1}
            label.optimized_thresholds['precision'] = {'score': best_precision, 'threshold': best_threshold_precision}
            label.optimized_thresholds['recall'] = {'score': best_recall, 'threshold': best_threshold_recall}
            self.label_thresholds[label.id_]['f1'] = {'score': best_f1, 'threshold': best_threshold_f1}
            self.label_thresholds[label.id_]['precision'] = {
                'score': best_precision,
                'threshold': best_threshold_precision,
            }
            self.label_thresholds[label.id_]['recall'] = {'score': best_recall, 'threshold': best_threshold_recall}

    def _query(self, search=None):
        """Query the comparison data.

        :param search: use a search query in pandas
        """
        from konfuzio_sdk.data import Document, Label, LabelSet

        if search is None:
            return self.data
        elif sdk_isinstance(search, Label):
            assert search.id_ is not None, f'{search} must have a ID'
            query = f'label_id == {search.id_} | (label_id_predicted == {search.id_})'
        elif sdk_isinstance(search, Document):
            assert search.id_ is not None, f'{search} must have a ID.'
            query = f'document_id == {search.id_} | (document_id_predicted == {search.id_})'
        elif sdk_isinstance(search, LabelSet):
            assert search.id_ is not None, f'{search} must have a ID.'
            query = f'label_set_id == {search.id_} | (label_set_id_predicted == {search.id_})'
        else:
            raise NotImplementedError
        return self.data.query(query)

    def tp(self, search=None) -> int:
        """Return the True Positives of all Spans."""
        return self._query(search=search)['true_positive'].sum()

    def fp(self, search=None) -> int:
        """Return the False Positives of all Spans."""
        return self._query(search=search)['false_positive'].sum()

    def fn(self, search=None) -> int:
        """Return the False Negatives of all Spans."""
        return self._query(search=search)['false_negative'].sum()

    def tn(self, search=None) -> int:
        """Return the True Negatives of all Spans."""
        return len(self._query(search=None)) - self.tp(search=search) - self.fn(search=search) - self.fp(search=search)

    def gt(self, search=None) -> int:
        """Return the number of ground-truth Annotations for a given Label."""
        return len(self._query(search=search).dropna(subset=['label_id']))

    def tokenizer_tp(self, search=None) -> int:
        """Return the tokenizer True Positives of all Spans."""
        return self._query(search=search)['tokenizer_true_positive'].sum()

    def tokenizer_fp(self, search=None) -> int:
        """Return the tokenizer False Positives of all Spans."""
        return self._query(search=search)['tokenizer_false_positive'].sum()

    def tokenizer_fn(self, search=None) -> int:
        """Return the tokenizer False Negatives of all Spans."""
        return self._query(search=search)['tokenizer_false_negative'].sum()

    def clf_tp(self, search=None) -> int:
        """Return the Label classifier True Positives of all Spans."""
        return self._query(search=search)['clf_true_positive'].sum()

    def clf_fp(self, search=None) -> int:
        """Return the Label classifier False Positives of all Spans."""
        return self._query(search=search)['clf_false_positive'].sum()

    def clf_fn(self, search=None) -> int:
        """Return the Label classifier False Negatives of all Spans."""
        return self._query(search=search)['clf_false_negative'].sum()

    def get_evaluation_data(self, search, allow_zero: bool = True) -> EvaluationCalculator:
        """Get precision, recall, f1, based on TP, FP, FN."""
        return EvaluationCalculator(
            tp=self.tp(search),
            fp=self.fp(search),
            fn=self.fn(search),
            tn=self.tn(search),
            zero_division=self.zero_division,
        )

    def precision(self, search=None) -> Optional[float]:
        """Calculate the Precision and see f1 to calculate imbalanced classes."""
        return EvaluationCalculator(
            tp=self.tp(search=search), fp=self.fp(search=search), zero_division=self.zero_division
        ).precision

    def recall(self, search=None) -> Optional[float]:
        """Calculate the Recall and see f1 to calculate imbalanced classes."""
        return EvaluationCalculator(
            tp=self.tp(search=search), fn=self.fn(search=search), zero_division=self.zero_division
        ).recall

    def f1(self, search=None) -> Optional[float]:
        """Calculate the F1 Score of one class.

        Please note: As suggested by Opitz et al. (2021) use the arithmetic mean over individual F1 scores.

        "F1 is often used with the intention to assign equal weight to frequent and infrequent classes, we recommend
        evaluating classifiers with F1 (the arithmetic mean over individual F1 scores), which is significantly more
        robust towards the error type distribution."

        Opitz, Juri, and Sebastian Burst. “Macro F1 and Macro F1.” arXiv preprint arXiv:1911.03347 (2021).
        https://arxiv.org/pdf/1911.03347.pdf

        :param search: Parameter used to calculate the value for one class.

        Example:
            1. If you have three Documents, calculate the F-1 Score per Document and use the arithmetic mean.
            2. If you have three Labels, calculate the F-1 Score per Label and use the arithmetic mean.
            3. If you have three Labels and three documents, calculate six F-1 Scores and use the arithmetic mean.

        """
        return EvaluationCalculator(
            tp=self.tp(search=search),
            fp=self.fp(search=search),
            fn=self.fn(search=search),
            zero_division=self.zero_division,
        ).f1

    def tokenizer_f1(self, search=None) -> Optional[float]:
        """
        Calculate the F1 Score of one the tokenizer.

        :param search: Parameter used to calculate the value for one Data object.
        """
        return EvaluationCalculator(
            tp=self.tokenizer_tp(search=search),
            fp=self.tokenizer_fp(search=search),
            fn=self.tokenizer_fn(search=search),
            zero_division=self.zero_division,
        ).f1

    def clf_f1(self, search=None) -> Optional[float]:
        """
        Calculate the F1 Score of one the Label classifier.

        :param search: Parameter used to calculate the value for one Data object.
        """
        return EvaluationCalculator(
            tp=self.clf_tp(search=search),
            fp=self.clf_fp(search=search),
            fn=self.clf_fn(search=search),
            zero_division=self.zero_division,
        ).f1

    def _apply(self, group, issue_name):
        """Vertical merge error methods helper method."""
        if len(group) < 2:
            group[issue_name] = False
            return group
        if len(set(group['id_local'])) > 1 or len(set(group['id_local_predicted'])) > 1:
            group[issue_name] = True
        return True

    def get_missing_vertical_merge(self):
        """Return Spans that should have been merged."""
        self.data.groupby('id_local').apply(lambda group: self._apply(group, 'missing_merge'))

        return self.data[self.data['missing_merge']]

    def get_wrong_vertical_merge(self):
        """Return Spans that were wrongly merged vertically."""
        self.data.groupby('id_local_predicted').apply(lambda group: self._apply(group, 'wrong_merge'))
        return self.data[self.data['wrong_merge']]

    def confusion_matrix(self):
        return ExtractionConfusionMatrix(data=self.data)


class CategorizationEvaluation:
    """Calculated evaluation measures for the classification task of Document categorization."""

    def __init__(self, categories: List[Category], documents: List[Tuple[Document, Document]], zero_division='warn'):
        """
        Relate to the two document instances.

        :param categories: The Categories to be evaluated.
        :param documents: A list of tuple Documents that should be compared.
        :param zero_division: Defines how to handle situations when precision, recall or F1 measure calculations result
        in zero division.
        Possible values: 'warn' – log a warning and assign a calculated metric a value of 0.
        0 - assign a calculated metric a value of 0.
        'error' – raise a ZeroDivisionError.
        None – assign None to a calculated metric.
        """
        self.categories = categories
        self.documents = documents
        self._evaluation_results = None
        self._clf_report = None
        self.zero_division = zero_division
        self.calculate()

    @property
    def category_ids(self) -> List[int]:
        """List of Category IDs as class labels."""
        return [category.id_ for category in self.categories]

    @property
    def category_names(self) -> List[str]:
        """List of Category names as class names."""
        return [category.name for category in self.categories]

    @property
    def actual_classes(self) -> List[int]:
        """List of ground truth Category IDs."""
        return [ground_truth.category.id_ for ground_truth, predicted in self.documents]

    @property
    def predicted_classes(self) -> List[int]:
        """List of predicted Category IDs."""
        return [
            predicted.category.id_ if predicted.category is not None else -1
            for ground_truth, predicted in self.documents
        ]

    def confusion_matrix(self) -> pd.DataFrame:
        """Confusion matrix."""
        return confusion_matrix(self.actual_classes, self.predicted_classes, labels=self.category_ids + [0])

    def _get_tp_tn_fp_fn_per_category(self) -> Dict[int, EvaluationCalculator]:
        """
        Get the TP, FP, TN and FN for each Category.

        The Category for which the evaluation is being done is considered the positive class. All others are considered
        as negative class.

        Follows the logic:
        tpi = cii (value in the diagonal of the cm for the respective Category)
        fpi = ∑nl=1 cli − tpi (sum of the column of the cm - except tp)
        fni = ∑nl=1 cil − tpi (sum of the row of the cm - except tp)
        tni = ∑nl=1 ∑nk=1 clk − tpi − fpi − fni (all other values not considered above)

        cm = [[1, 1, 0],
            [0, 2, 1],
            [1, 2, 3]]

        For Category '1':
        tp = 2
        fp = 1 + 2 = 3
        fn = 1 + 0 = 1
        tn = 11 - 2 - 3 - 1 = 5

        :return: dictionary with the results per Category
        """
        confusion_matrix = self.confusion_matrix()
        sum_columns = np.sum(confusion_matrix, axis=0)
        sum_rows = np.sum(confusion_matrix, axis=1)
        sum_all = np.sum(confusion_matrix)

        results = {}

        for ind, category_id in enumerate(self.category_ids):
            tp = confusion_matrix[ind, ind]
            fp = sum_columns[ind] - tp
            fn = sum_rows[ind] - tp
            tn = sum_all - fn - fp - tp
            results[category_id] = EvaluationCalculator(
                tp=tp, fp=fp, fn=fn, tn=tn, zero_division=self.zero_division
            )  # the value is evaluation calculator, not a tuple as a result

        return results

    def _get_tp_tn_fp_fn_across_categories(self) -> EvaluationCalculator:
        """Get the TP, FP, TN and FN across all Categories."""
        result = classification_report(
            y_true=self.actual_classes,
            y_pred=self.predicted_classes,
            labels=self.category_ids,
            target_names=self.category_names,
            output_dict=True,
        )['weighted avg']
        result['f1'] = result['f1-score']
        return result

    def calculate(self):
        """Calculate and update the data stored within this Evaluation."""
        self._evaluation_results = self._get_tp_tn_fp_fn_per_category()
        self._clf_report = self._get_tp_tn_fp_fn_across_categories()

    def _base_metric(self, metric: str, category: Optional[Category] = None) -> int:
        """Return the base metric of all Documents filtered by Category.

        :param metric: One of TP, FP, FN, TN.
        :param category: A Category to filter for, or None for getting global evaluation results.
        """
        return sum(
            [
                getattr(evaluation, metric)
                for category_id, evaluation in self._evaluation_results.items()
                if (category is None) or (category_id == category.id_)
            ]
        )

    def tp(self, category: Optional[Category] = None) -> int:
        """Return the True Positives of all Documents."""
        return self._base_metric('tp', category)

    def fp(self, category: Optional[Category] = None) -> int:
        """Return the False Positives of all Documents."""
        return self._base_metric('fp', category)

    def fn(self, category: Optional[Category] = None) -> int:
        """Return the False Negatives of all Documents."""
        return self._base_metric('fn', category)

    def tn(self, category: Optional[Category] = None) -> int:
        """Return the True Negatives of all Documents."""
        return self._base_metric('tn', category)

    def gt(self, category: Optional[Category] = None) -> int:
        """Placeholder for compatibility with Server."""
        return 0

    def get_evaluation_data(self, search: Category = None, allow_zero: bool = True) -> EvaluationCalculator:
        """
        Get precision, recall, f1, based on TP, TN, FP, FN.

        :param search: A Category to filter for, or None for getting global evaluation results.
        :type search: Category
        :param allow_zero: If true, will calculate None for precision and recall when the straightforward application
        of the formula would otherwise result in 0/0. Raises ZeroDivisionError otherwise.
        :type allow_zero: bool
        """
        return EvaluationCalculator(
            tp=self.tp(search),
            fp=self.fp(search),
            fn=self.fn(search),
            tn=self.tn(search),
            zero_division=self.zero_division,
        )

    def _metric(self, metric: str, category: Optional[Category]) -> Optional[float]:
        """Calculate a global metric or filter it by one Category.

        :param metric: One of precision, recall, or f1.
        :param category: A Category to filter for, or None for getting global evaluation results.
        """
        metric = metric.lower()
        if metric not in ['precision', 'recall', 'f1']:
            raise NotImplementedError
        if category is None:
            return self._clf_report[metric]
        else:
            return getattr(self.get_evaluation_data(search=category), metric)

    def precision(self, category: Optional[Category]) -> Optional[float]:
        """Calculate the global Precision or filter it by one Category."""
        return self._metric('precision', category)

    def recall(self, category: Optional[Category]) -> Optional[float]:
        """Calculate the global Recall or filter it by one Category."""
        return self._metric('recall', category)

    def f1(self, category: Optional[Category]) -> Optional[float]:
        """Calculate the global F1 Score or filter it by one Category."""
        return self._metric('f1', category)


class FileSplittingEvaluation:
    """Evaluate the quality of the filesplitting logic."""

    def __init__(
        self,
        ground_truth_documents: List[Document],
        prediction_documents: List[Document],
        zero_division='warn',
    ):
        """
        Initialize and run the metrics calculation.

        :param ground_truth_documents: A list of original unchanged Documents.
        :type ground_truth_documents: list
        :param prediction_documents: A list of Documents with Pages newly predicted to be first or non-first.
        :type prediction_documents: list
        :param zero_division: Defines how to handle situations when precision, recall or F1 measure calculations result
        in zero division.
        Possible values: 'warn' – log a warning and assign a calculated metric a value of 0.
        0 - assign a calculated metric a value of 0.
        'error' – raise a ZeroDivisionError.
        None – assign None to a calculated metric.
        :raises ValueError: When ground_truth_documents and prediction_documents are not the same length.
        :raises ValueError: When a Page does not have a value of is_first_page.
        :raises ValueError: When an original Document and prediction are not referring to the same Document.

        """
        if len(ground_truth_documents) != len(prediction_documents):
            raise ValueError('ground_truth_documents and prediction_documents must be same length.')
        for document in ground_truth_documents:
            for page in document.pages():
                if page.is_first_page is None:
                    raise ValueError(f'Page {page.number} of {document} does not have a value of is_first_page.')
        for document in prediction_documents:
            for page in document.pages():
                if page.is_first_page is None:
                    raise ValueError(
                        f'Page {page.number} of prediction of {document.copy_of_id} does not have a value '
                        f'of is_first_page.'
                    )
        for ground_truth, prediction in zip(ground_truth_documents, prediction_documents):
            if ground_truth.id_ not in [prediction.copy_of_id, prediction.id_]:
                raise ValueError(
                    f'Incorrect prediction passed for {ground_truth}. Prediction has to be a copy of a '
                    f'ground truth Document.'
                )
        projects = list({document.project for document in ground_truth_documents})
        if len(projects) > 1:
            raise ValueError('All Documents have to belong to the same Project.')
        print(
            f'ground_truth_documents: {len(ground_truth_documents)}, prediction_documents: {len(prediction_documents)}'
        )
        self.document_pairs = [
            [document[0], document[1]] for document in zip(ground_truth_documents, prediction_documents)
        ]
        print(f'project:{projects}')
        self.project = projects[0]  # because we check that exactly one Project exists across the Documents
        self.zero_division = zero_division
        self.evaluation_results = None
        self.evaluation_results_by_category = None
        self.calculate()
        self.calculate_metrics_by_category()

    def _metric_calculations(self, category=None):
        """
        Calculate metrics for a single category.

        :param category: A Category to calculate metrics for.
        :type category: Category
        :returns: Seven metrics.
        """
        tp, fp, fn, tn = 0, 0, 0, 0
        if category:
            evaluation_documents = [
                [document_1, document_2]
                for document_1, document_2 in self.document_pairs
                if document_1.category and document_1.category.id_ == category.id_
            ]
        else:
            evaluation_documents = self.document_pairs
        for ground_truth, prediction in evaluation_documents:
            print(f'gr_id: {ground_truth.id_}, pr_id: {prediction.id_}, pages_len: {len(ground_truth.pages())}')
            for page_gt, page_pr in zip(ground_truth.pages(), prediction.pages()):
                print(
                    f'gt: {page_gt.is_first_page}, pr: {page_pr.is_first_page}, pr_confidence: {page_pr.is_first_page_confidence:.2f}, is_correct: {page_gt.is_first_page == page_pr.is_first_page}, document: {page_gt.document.id_}'
                )
                if page_gt.is_first_page and page_pr.is_first_page:
                    tp += 1
                elif not page_gt.is_first_page and page_pr.is_first_page:
                    fp += 1
                elif page_gt.is_first_page and not page_pr.is_first_page:
                    fn += 1
                elif not page_gt.is_first_page and not page_pr.is_first_page:
                    tn += 1
        evaluation_calculator = EvaluationCalculator(tp=tp, fp=fp, fn=fn, tn=tn, zero_division=self.zero_division)
        precision = evaluation_calculator.precision
        recall = evaluation_calculator.recall
        f1 = evaluation_calculator.f1
        return tp, fp, fn, tn, precision, recall, f1

    def calculate(self):
        """Calculate metrics for the File Splitting logic."""
        tp, fp, fn, tn, precision, recall, f1 = self._metric_calculations()
        self.evaluation_results = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def calculate_metrics_by_category(self):
        """Calculate metrics by Category independently."""
        categories = list({doc_pair[0].category for doc_pair in self.document_pairs})
        self.evaluation_results_by_category = {}
        for category in categories:
            self.evaluation_results_by_category[category.id_] = {}
            tp, fp, fn, tn, precision, recall, f1 = self._metric_calculations(category)
            self.evaluation_results_by_category[category.id_]['true_positives'] = tp
            self.evaluation_results_by_category[category.id_]['false_positives'] = fp
            self.evaluation_results_by_category[category.id_]['false_negatives'] = fn
            self.evaluation_results_by_category[category.id_]['true_negatives'] = tn
            self.evaluation_results_by_category[category.id_]['precision'] = precision
            self.evaluation_results_by_category[category.id_]['recall'] = recall
            self.evaluation_results_by_category[category.id_]['f1'] = f1

    def _query(self, metric: str, search: Category = None) -> Union[int, float, None]:
        """
        Get a specific metric for a given category or get all metrics for all categories.

        :param metric: The name of the metric to get.
        :type metric: str
        :param search: The Category to get the metric for, if not provided will return all metrics for all Categories.
        :type search: Category
        :returns: A metric or a dictionary of metrics.
        :raises KeyError: If the given Category is not present in the project the evaluation is running on.
        """
        if search:
            if search.id_ not in self.evaluation_results_by_category:
                raise KeyError(
                    f'{search} is not present in {self.project}. Only Categories within a Project can be used for '
                    f'viewing metrics.'
                )
            return self.evaluation_results_by_category[search.id_][metric]
        return self.evaluation_results[metric]

    def get_evaluation_data(self, search: Category = None, allow_zero: bool = True) -> EvaluationCalculator:
        """
        Get precision, recall, f1, based on TP, TN, FP, FN.

        :param search: display true positives within a certain Category.
        :type search: Category
        :param allow_zero: If true, will calculate None for precision and recall when the straightforward application
        of the formula would otherwise result in 0/0. Raises ZeroDivisionError otherwise.
        :type allow_zero: bool
        """
        return EvaluationCalculator(
            tp=self.tp(search),
            fp=self.fp(search),
            fn=self.fn(search),
            tn=self.tn(search),
            zero_division=self.zero_division,
        )

    def tp(self, search: Category = None) -> int:
        """
        Return correctly predicted first Pages.

        :param search: display true positives within a certain Category.
        :type search: Category
        :raises KeyError: When the Category in search is not present in the Project from which the Documents are.
        """
        return self._query('true_positives', search)

    def fp(self, search: Category = None) -> int:
        """
        Return non-first Pages incorrectly predicted as first.

        :param search: display false positives within a certain Category.
        :type search: Category
        :raises KeyError: When the Category in search is not present in the Project from which the Documents are.
        """
        return self._query('false_positives', search)

    def fn(self, search: Category = None) -> int:
        """
        Return first Pages incorrectly predicted as non-first.

        :param search: display false negatives within a certain Category.
        :type search: Category
        :raises KeyError: When the Category in search is not present in the Project from which the Documents are.
        """
        return self._query('false_negatives', search)

    def tn(self, search: Category = None) -> int:
        """
        Return non-first Pages predicted as non-first.

        :param search: display true negatives within a certain Category.
        :type search: Category
        :raises KeyError: When the Category in search is not present in the Project from which the Documents are.
        """
        return self._query('true_negatives', search)

    def gt(self, search: Category = None) -> int:
        """Placeholder for compatibility with Server."""
        return 0

    def precision(self, search: Category = None) -> float:
        """
        Return precision.

        :param search: display precision within a certain Category.
        :type search: Category
        :raises KeyError: When the Category in search is not present in the Project from which the Documents are.
        """
        return self._query('precision', search)

    def recall(self, search: Category = None) -> float:
        """
        Return recall.

        :param search: display recall within a certain Category.
        :type search: Category
        :raises KeyError: When the Category in search is not present in the Project from which the Documents are.
        """
        return self._query('recall', search)

    def f1(self, search: Category = None) -> float:
        """
        Return F1-measure.

        :param search: display F1 measure within a certain Category.
        :type search: Category
        :raises KeyError: When the Category in search is not present in the Project from which the Documents are.
        """
        return self._query('f1', search)
