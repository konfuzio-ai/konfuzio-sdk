"""Test code examples for evaluation of the Extraction AI in the documentation."""
from konfuzio_sdk.trainer.information_extraction import load_model

MODEL_PATH = 'docs/sdk/boilerplates/2023-02-25-22-39-04_lohnabrechnung.pkl'
pipeline = load_model(MODEL_PATH)

# To get the evaluation of the full pipeline
evaluation = pipeline.evaluate_full()
assert evaluation.f1() > 0.7
print(f"Full evaluation F1 score: {evaluation.f1()}")
assert evaluation.recall() == 0.75
print(f"Full evaluation recall: {evaluation.recall()}")
assert evaluation.precision() > 0.7
print(f"Full evaluation precision: {evaluation.precision()}")

# To get the evaluation of the Tokenizer alone
evaluation = pipeline.evaluate_tokenizer()
assert evaluation.tokenizer_f1() > 0.5
print(f"Tokenizer evaluation F1 score: {evaluation.tokenizer_f1()}")

# To get the evaluation of the Label classifier given perfect tokenization
evaluation = pipeline.evaluate_clf()
assert evaluation.clf_f1() > 0.95
print(f"Label classifier evaluation F1 score: {evaluation.clf_f1()}")

# To get the evaluation of the LabelSet given perfect Label classification
evaluation = pipeline.evaluate_clf()
assert evaluation.f1() > 0.9
print(f"Label Set evaluation F1 score: {evaluation.f1()}")
