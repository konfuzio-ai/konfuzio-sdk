.. meta::
   :description: Docstrings of the classes, methods, and functions in the source code of the konfuzio-sdk package with references to the code itself.


=====================
API Reference
=====================

*Reference guides are technical descriptions of the machinery and how to operate it.*
*Reference material is information-oriented.*


Data
=====================

`[source] <https://github.com/konfuzio-ai/document-ai-python-sdk/blob/master/konfuzio_sdk/data.py>`__

.. automodule:: konfuzio_sdk.data


Span
---------------------

.. autoclass:: Span
   :members:
   :noindex:


Bbox
---------------------

.. autoclass:: Bbox
   :members:
   :noindex:

Annotation
---------------------
.. autoclass:: Annotation
   :members:
   :noindex:


Annotation Set
---------------------
.. autoclass:: AnnotationSet
   :members:
   :noindex:


Label
---------------------

.. autoclass:: Label
   :members:
   :noindex:


Label Set
---------------------
.. autoclass:: LabelSet
   :members:
   :noindex:


Category
---------------------
.. autoclass:: Category
   :members:
   :noindex:


Document
---------------------
.. autoclass:: Document
   :members:
   :noindex:


Page
---------------------
.. autoclass:: Page
   :members:
   :noindex:


Project
---------------------
.. autoclass:: Project
   :members:
   :noindex:


Tokenizers
=====================

`[source] <https://github.com/konfuzio-ai/document-ai-python-sdk/blob/master/konfuzio_sdk/tokenizer>`__

.. automodule:: konfuzio_sdk.tokenizer.base

Abstract Tokenizer
---------------------
.. autoclass:: AbstractTokenizer
   :members:
   :noindex:

List Tokenizer
---------------------
.. autoclass:: ListTokenizer
   :members:
   :noindex:

Rule Based Tokenizer
---------------------

.. automodule:: konfuzio_sdk.tokenizer.regex

Regex Tokenizer
---------------------
.. autoclass:: RegexTokenizer
   :members:
   :noindex:

.. automodule:: konfuzio_sdk.tokenizer.regex
    :members:

FileSplitting AI
=====================

`[source] <https://github.com/konfuzio-ai/konfuzio-sdk/blob/master/konfuzio_sdk/trainer/file_splitting.py>`__

.. automodule:: konfuzio_sdk.trainer.file_splitting

.. autoclass:: AbstractFileSplittingModel
   :members:
   :noindex:

.. autoclass:: ContextAwareFileSplittingModel
   :members:
   :noindex:

.. autoclass:: SplittingAI
   :members:
   :noindex:

Extraction AI
=====================

`[source] <https://github.com/konfuzio-ai/document-ai-python-sdk/blob/master/konfuzio_sdk/trainer/information_extraction.py>`__


.. automodule:: konfuzio_sdk.trainer.information_extraction

.. autoclass:: RFExtractionAI
   :members:
   :noindex:

Load Saved AI Model
---------------------

.. autofunction:: konfuzio_sdk.trainer.information_extraction.load_model
   :noindex:

Document Categorization
=====================

`[source] <https://github.com/konfuzio-ai/document-ai-python-sdk/blob/master/konfuzio_sdk/trainer/document_categorization.py>`__

.. automodule:: konfuzio_sdk.trainer.document_categorization


Fallback Categorization Model
---------------------
.. autoclass:: FallbackCategorizationModel
   :members:
   :noindex:


Evaluation
=====================

`[source] <https://github.com/konfuzio-ai/document-ai-python-sdk/blob/master/konfuzio_sdk/evaluate.py>`__

Extraction AI Evaluation
=====================
.. autoclass:: konfuzio_sdk.evaluate.Evaluation
   :members:
   :noindex:

compare
---------------------
.. autofunction:: konfuzio_sdk.evaluate.compare
   :noindex:

Categorization AI Evaluation
=====================
.. autoclass:: konfuzio_sdk.evaluate.CategorizationEvaluation
   :members:
   :noindex:
