## Document Information Extraction

### Train a Konfuzio SDK Model to Extract Information From Payslip Documents

The tutorial *RFExtractionAI Demo* aims to show you how to use the Konfuzio SDK package to use a simple `Whitespace
tokenizer <https://dev.konfuzio.com/sdk/sourcecode.html#konfuzio_sdk.tokenizer.regex.WhitespaceTokenizer>`_ and to
train a "RFExtractionAI" model to find and extract relevant information like Name, Date and Recipient
from payslip documents.

You can |OpenInColab| or download it from [here](https://github.com/konfuzio-ai/document-ai-python-sdk/blob/master/docs/sdk/examples/RFExtractionAI%20Demo.ipynb)
and try it by yourself.

.. |OpenInColab| image:: https://colab.research.google.com/assets/colab-badge.svg
.. _OpenInColab: https://colab.research.google.com/github/konfuzio-ai/document-ai-python-sdk/blob/master/docs/sdk/examples/RFExtractionAI%20Demo.ipynb

### Evaluate a Trained Extraction AI Model

In this example we will see how we can evaluate a trained `RFExtractionAI` model. We will assume that we have a trained 
pickled model available. See [here](https://dev.konfuzio.com/sdk/examples/examples.html#train-a-konfuzio-sdk-model-to-extract-information-from-payslip-documents) 
for how to train such a model, and check out the [Evaluation](https://dev.konfuzio.com/sdk/sourcecode.html#ai-evaluation) 
documentation for more details.

.. literalinclude:: /sdk/boilerplates/test_evaluate_extraction_ai.py
   :language: python
   :lines: 11-12,14-15,21-22,24,26,28-31,33-36,38-41,43
   :dedent: 4
