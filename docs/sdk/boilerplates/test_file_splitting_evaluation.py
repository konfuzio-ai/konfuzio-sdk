"""Test File Splitting evaluation documentation examples."""
from konfuzio_sdk.evaluate import FileSplittingEvaluation

YOUR_GROUND_TRUTH_LIST = []
YOUR_PREDICTION_LIST = []
YOUR_CATEGORY = None
evaluation = FileSplittingEvaluation(
    ground_truth_documents=YOUR_GROUND_TRUTH_LIST, prediction_documents=YOUR_PREDICTION_LIST
)

print(evaluation.fn())
print(evaluation.fn(search=YOUR_CATEGORY))
