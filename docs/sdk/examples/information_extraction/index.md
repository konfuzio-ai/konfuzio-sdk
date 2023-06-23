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

### Train a custom Extraction AI

This section explains how to train a custom Extraction AI locally, how to save it and upload it to the Konfuzio Server. 
To prepare the data for training and testing your AI, follow the [data preparation tutorial](tutorials.html#tutorials.html#prepare-the-data-for-training-and-testing-the-ai).

By default, any Extraction AI class should derive from the `AbstractExtractionAI` class and implement the following 
interface:

.. literalinclude:: /sdk/boilerplates/test_custom_extraction_ai.py
      :language: python
      :start-after: start custom
      :end-before: end custom
      :dedent: 4

Example usage of your Custom Extraction AI:

.. literalinclude:: /sdk/boilerplates/test_custom_extraction_ai.py
      :language: python
      :start-after: start init_ai
      :end-before: end init_ai
      :dedent: 4
.. literalinclude:: /sdk/boilerplates/test_custom_extraction_ai.py
      :language: python
      :start-after: start category
      :end-before: end category
      :dedent: 4
.. literalinclude:: /sdk/boilerplates/test_custom_extraction_ai.py
      :language: python
      :start-after: start train
      :end-before: end train
      :dedent: 4

The custom AI inherits from [BaseModel](sourcecode.html#base-model), which provides `BaseModel.save` to generate a 
pickle file that can be directly uploaded to the Konfuzio Server (see [Upload Extraction or Category AI to target instance](https://help.konfuzio.com/tutorials/migrate-trained-ai-to-an-new-project-to-annotate-documents-faster/index.html#upload-extraction-or-category-ai-to-target-instance)). 

Activating the uploaded AI on the web interface will enable the custom pipeline on your self-hosted installation.

It is also possible to upload the AI from your local machine using the `upload_ai_model()` method and remove it with the
`delete_ai_model()` method:

```python
from konfuzio_sdk.api import upload_ai_model, delete_ai_model

# upload a saved model to the server
model_id = upload_ai_model(pickle_model_path)

# remove model
delete_ai_model(model_id, ai_type='extraction')
```

### Evaluate a Trained Extraction AI Model

In this example we will see how we can evaluate a trained `RFExtractionAI` model. We will assume that we have a trained 
pickled model available. Check out the [Evaluation](https://dev.konfuzio.com/sdk/sourcecode.html#ai-evaluation) 
documentation for more details.

.. literalinclude:: /sdk/boilerplates/test_evaluate_extraction_ai.py
   :language: python
   :start-after: start init
   :end-before: end init
   :dedent: 4
.. literalinclude:: /sdk/boilerplates/test_evaluate_extraction_ai.py
   :language: python
   :start-after: start scores
   :end-before: end scores
   :dedent: 4
