"""Test File Splitting evaluation documentation examples."""
import pytest

from konfuzio_sdk.settings_importer import is_dependency_installed


@pytest.mark.skipif(
    not is_dependency_installed('torch')
    and not is_dependency_installed('transformers')
    and not is_dependency_installed('tensorflow')
    and not is_dependency_installed('cloudpickle'),
    reason='Required dependencies not installed.',
)
def test_file_splitting_evaluation():
    """Test File Splitting evaluation."""
    from konfuzio_sdk.samples import LocalTextProject
    from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
    from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel

    YOUR_PROJECT = LocalTextProject()
    YOUR_CATEGORY_1 = YOUR_PROJECT.get_category_by_id(3)
    YOUR_CATEGORY_2 = YOUR_PROJECT.get_category_by_id(4)

    # start document creation
    from konfuzio_sdk.data import Document, Page
    from konfuzio_sdk.evaluate import FileSplittingEvaluation, EvaluationCalculator
    from konfuzio_sdk.trainer.file_splitting import SplittingAI

    # This example builds the Documents from scratch and without uploading a Supported File.
    # If you uploaded your Document to the Konfuzio Server, you can just retrieve it with:
    # document_1 = project.get_document_by_id(YOUR_DOCUMENT_ID)
    text_1 = "Hi all,\nI like bread.\nI hope to get everything done soon.\nHave you seen it?"
    document_1 = Document(id_=20, project=YOUR_PROJECT, category=YOUR_CATEGORY_1, text=text_1, dataset_status=3)
    _ = Page(
        id_=None, original_size=(320, 240), document=document_1, start_offset=0, end_offset=21, number=1, copy_of_id=29
    )

    _ = Page(
        id_=None, original_size=(320, 240), document=document_1, start_offset=22, end_offset=57, number=2, copy_of_id=30
    )

    _ = Page(
        id_=None, original_size=(320, 240), document=document_1, start_offset=58, end_offset=75, number=3, copy_of_id=31
    )

    # As with the previous example Document, you can just retrieve an online Document with
    # document_2 = project.get_document_by_id(YOUR_DOCUMENT_ID)
    text_2 = "Evening,\nthank you for coming.\nI like fish.\nI need it.\nEvening."
    document_2 = Document(id_=21, project=YOUR_PROJECT, category=YOUR_CATEGORY_2, text=text_2, dataset_status=3)
    _ = Page(
        id_=None, original_size=(320, 240), document=document_2, start_offset=0, end_offset=8, number=1, copy_of_id=32
    )
    _ = Page(
        id_=None, original_size=(320, 240), document=document_2, start_offset=9, end_offset=30, number=2, copy_of_id=33
    )
    _ = Page(
        id_=None, original_size=(320, 240), document=document_2, start_offset=31, end_offset=43, number=3, copy_of_id=34
    )
    _.is_first_page = True
    _ = Page(
        id_=None, original_size=(320, 240), document=document_2, start_offset=44, end_offset=54, number=4, copy_of_id=35
    )
    _ = Page(
        id_=None, original_size=(320, 240), document=document_2, start_offset=55, end_offset=63, number=5, copy_of_id=36
    )
    _.is_first_page = True
    # end document creation

    assert len(document_1.pages()) == 3
    assert len(document_2.pages()) == 5

    YOUR_MODEL = ContextAwareFileSplittingModel(
        categories=[YOUR_CATEGORY_1, YOUR_CATEGORY_2], tokenizer=ConnectedTextTokenizer()
    )
    YOUR_MODEL.fit()

    # start splitting
    splitting_ai = SplittingAI(YOUR_MODEL)
    pred_1: Document = splitting_ai.propose_split_documents(document_1, return_pages=True)[0]
    pred_2: Document = splitting_ai.propose_split_documents(document_2, return_pages=True)[0]

    evaluation = FileSplittingEvaluation(
        ground_truth_documents=[document_1, document_2], prediction_documents=[pred_1, pred_2]
    )
    # end splitting

    YOUR_GROUND_TRUTH_LIST = [document_1, document_2]
    YOUR_PREDICTION_LIST = [pred_1, pred_2]
    YOUR_CATEGORY = YOUR_CATEGORY_1
    # start eval_example
    evaluation = FileSplittingEvaluation(
        ground_truth_documents=YOUR_GROUND_TRUTH_LIST, prediction_documents=YOUR_PREDICTION_LIST
    )
    # end eval_example

    assert evaluation.tp() == 3
    assert evaluation.tn() == 4
    assert evaluation.fp() == 0
    assert evaluation.fn() == 1
    assert evaluation.precision() == 1
    assert evaluation.recall() == 0.75
    assert evaluation.f1() == 0.8571428571428571

    # start scores
    print(evaluation.tp())
    # returns: 3
    print(evaluation.tn())
    # returns: 4
    print(evaluation.fp())
    # returns: 0
    print(evaluation.fn())
    # returns: 1
    print(evaluation.precision())
    # returns: 1
    print(evaluation.recall())
    # returns: 0.75
    print(evaluation.f1())
    # returns: 0.85
    # end scores

    assert evaluation.tp(search=YOUR_CATEGORY_1) == 1
    assert evaluation.tp(search=YOUR_CATEGORY_2) == 2
    assert evaluation.tn(search=YOUR_CATEGORY_1) == 2
    assert evaluation.tn(search=YOUR_CATEGORY_2) == 2
    assert evaluation.fp(search=YOUR_CATEGORY_1) == 0
    assert evaluation.fp(search=YOUR_CATEGORY_2) == 0
    assert evaluation.fn(search=YOUR_CATEGORY_1) == 0
    assert evaluation.fn(search=YOUR_CATEGORY_2) == 1
    assert evaluation.precision(search=YOUR_CATEGORY_1) == 1
    assert evaluation.precision(search=YOUR_CATEGORY_2) == 1
    assert evaluation.recall(search=YOUR_CATEGORY_1) == 1
    assert evaluation.recall(search=YOUR_CATEGORY_2) == 0.6666666666666666
    assert evaluation.f1(search=YOUR_CATEGORY_1) == 1
    assert evaluation.f1(search=YOUR_CATEGORY_2) == 0.8

    # start scores_category
    print(evaluation.tp(search=YOUR_CATEGORY_1), evaluation.tp(search=YOUR_CATEGORY_2))
    # returns: 1 2
    print(evaluation.tn(search=YOUR_CATEGORY_1), evaluation.tn(search=YOUR_CATEGORY_2))
    # returns: 2 2
    print(evaluation.fp(search=YOUR_CATEGORY_1), evaluation.fp(search=YOUR_CATEGORY_2))
    # returns: 0 0
    print(evaluation.fn(search=YOUR_CATEGORY_1), evaluation.fn(search=YOUR_CATEGORY_2))
    # returns: 0 1
    print(evaluation.precision(search=YOUR_CATEGORY_1), evaluation.precision(search=YOUR_CATEGORY_2))
    # returns: 1 1
    print(evaluation.recall(search=YOUR_CATEGORY_1), evaluation.recall(search=YOUR_CATEGORY_2))
    # returns: 1 0.66
    print(evaluation.f1(search=YOUR_CATEGORY_1), evaluation.f1(search=YOUR_CATEGORY_2))
    # returns: 1 0.8
    # end scores_category

    # start calculator
    EvaluationCalculator(tp=3, fp=0, fn=1, tn=4).metrics_logging()
    # end calculator

    # start single_metric
    print(evaluation.fn())
    # end single_metric

    # start metric_category
    print(evaluation.fn(search=YOUR_CATEGORY))
    # end metric_category
