## Document Information Extraction

### Train a Konfuzio SDK Model to Extract Information From Payslip Documents

.. _Information Extraction:

The tutorial *RFExtractionAI Demo* aims to show you how to use the Konfuzio SDK package to use a simple `Whitespace
tokenizer <https://dev.konfuzio.com/sdk/sourcecode.html#konfuzio_sdk.tokenizer.regex.WhitespaceTokenizer>`_ and to
train a "RFExtractionAI" model to find and extract relevant information like Name, Date and Recipient
from payslip documents.

You can <a href="https://colab.research.google.com/github/konfuzio-ai/document-ai-python-sdk/blob/master/docs/sdk/tutorials/RFExtractionAI%20Demo.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> or download it from [here](https://github.com/konfuzio-ai/document-ai-python-sdk/blob/master/docs/sdk/tutorials/RFExtractionAI%20Demo.ipynb)
and try it by yourself.

.. |OpenInColab| image:: https://colab.research.google.com/assets/colab-badge.svg
.. _OpenInColab: https://colab.research.google.com/github/konfuzio-ai/document-ai-python-sdk/blob/master/docs/sdk/tutorials/RFExtractionAI%20Demo.ipynb

### Customize Extraction AI

Any Custom [Extraction AI](sourcecode.html#extraction-ai) (derived from the Konfuzio `AbstractExtractionAI` class) should implement 
the following interface:

```python
from konfuzio_sdk.trainer.information_extraction import AbstractExtractionAI
from konfuzio_sdk.data import Document
from typing import List


class CustomExtractionAI(AbstractExtractionAI):

    def __init__(self, *args, **kwargs):

    # initialize key variables required by the custom AI

    def fit(self):

    # Define architecture and training that the model undergoes, i.e. a NN architecture or a custom hardcoded logic
    # This method is allowed to be implemented as a no-op if you provide the trained model in other ways

    def extract(self, document: Document) -> Document:
# define how the model extracts information from Documents
# **NB:** The result of extraction must be a copy of the input Document with added Annotations attribute `Document._annotations`
```

Example usage of your Custom Extraction AI:
```python
from konfuzio_sdk.data import Project, Document
from konfuzio_sdk.trainer.information_extraction import load_model

# Initialize Project and provide the AI training and test data
project = Project(id_=YOUR_PROJECT_ID)  # see https://dev.konfuzio.com/sdk/get_started.html#example-usage

extraction_pipeline = CustomExtractionAI(*args, **kwargs)
extraction_pipeline.category = project.get_category_by_id(id_=YOUR_CATEGORY_ID)
extraction_pipeline.documents = extraction_pipeline.category.documents()
extraction_pipeline.test_documents = extraction_pipeline.category.test_documents()

# Train the AI
extraction_pipeline.fit()

# Evaluate the AI
data_quality = extraction_pipeline.evaluate_full(use_training_docs=True)
ai_quality = extraction_pipeline.evaluate_full(use_training_docs=False)

# Extract a Document
document = self.project.get_document_by_id(YOUR_DOCUMENT_ID)
extraction_result: Document = extraction_pipeline.extract(document=document)

# Save and load a pickle file for the model
pickle_model_path = extraction_pipeline.save(output_dir=project.model_folder, include_konfuzio=True)
extraction_pipeline_loaded = load_model(pickle_model_path)
```

### Evaluate a Trained Extraction AI Model

In this example we will see how we can evaluate a trained `RFExtractionAI` model. We will assume that we have a trained 
pickled model available. See :ref:`here <Information Extraction>` 
for how to train such a model, and check out the [Evaluation](https://dev.konfuzio.com/sdk/sourcecode.html#ai-evaluation) 
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
