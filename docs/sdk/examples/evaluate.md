
## Evaluate a Trained Extraction AI Model

In this example we will see how we can evaluate a trained `RFExtractionAI` model. We will assume that we have a trained pickled model available. See [here](https://dev.konfuzio.com/sdk/examples/examples.html#train-a-konfuzio-sdk-model-to-extract-information-from-payslip-documents) for how to train such a model, and check out the [Evaluation](https://dev.konfuzio.com/sdk/sourcecode.html#evaluation) documentation for more details.

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.information_extraction import load_model

pipeline = load_model(MODEL_PATH)

# To get the evaluation of the full pipeline
evaluation = pipeline.evaluate_full()
print(f"Full evaluation F1 score: {evaluation.f1()}")
print(f"Full evaluation recall: {evaluation.recall()}")
print(f"Full evaluation precision: {evaluation.precision()}")

# To get the evaluation of the tokenizer alone
evaluation = pipeline.evaluate_tokenizer()
print(f"Tokenizer evaluation F1 score: {evaluation.tokenizer_f1()}")

# To get the evaluation of the Label classifier given perfect tokenization
evaluation = pipeline.evaluate_clf()
print(f"Label classifier evaluation F1 score: {evaluation.clf_f1()}")

# To get the evaluation of the LabelSet given perfect Label classification
evaluation = pipeline.evaluate_template_clf()
print(f"Label Set evaluation F1 score: {evaluation.f1()}")

```