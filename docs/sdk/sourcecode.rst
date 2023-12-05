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


Category Annotation
---------------------
.. autoclass:: CategoryAnnotation
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

.. automodule:: konfuzio_sdk.tokenizer.paragraph_and_sentence

Paragraph Tokenizer
---------------------
.. autoclass:: ParagraphTokenizer
   :members:
   :noindex:

Sentence Tokenizer
---------------------
.. autoclass:: SentenceTokenizer
   :members:
   :noindex:

Extraction AI
=====================

`[source] <https://github.com/konfuzio-ai/document-ai-python-sdk/blob/master/konfuzio_sdk/trainer/information_extraction.py>`__

.. automodule:: konfuzio_sdk.trainer.information_extraction

Base Model
---------------------

.. autoclass:: BaseModel
   :members:
   :noindex:

AbstractExtractionAI
---------------------

.. autoclass:: AbstractExtractionAI
   :members:
   :noindex:

Random Forest Extraction AI
---------------------

.. autoclass:: RFExtractionAI
   :members:
   :noindex:


Categorization AI
=====================

`[source] <https://github.com/konfuzio-ai/document-ai-python-sdk/blob/master/konfuzio_sdk/trainer/document_categorization.py>`__

.. automodule:: konfuzio_sdk.trainer.document_categorization

Abstract Categorization AI
---------------------
.. autoclass:: AbstractCategorizationAI
   :members:
   :noindex:

Name-based Categorization AI
---------------------
.. autoclass:: NameBasedCategorizationAI
   :members:
   :noindex:

Model-based Categorization AI
---------------------
.. autoclass:: CategorizationAI
   :members:
   :noindex:

Build a Model-based Categorization AI
---------------------
.. autofunction:: build_categorization_ai_pipeline
   :noindex:

NBOW Model
---------------------
.. autoclass:: NBOW
   :members:
   :noindex:

NBOW Self Attention Model
---------------------
.. autoclass:: NBOWSelfAttention
   :members:
   :noindex:

LSTM Model
---------------------
.. autoclass:: LSTM
   :members:
   :noindex:

BERT Model
---------------------
.. autoclass:: BERT
   :members:
   :noindex:

VGG Model
---------------------
.. autoclass:: VGG
   :members:
   :noindex:

EfficientNet Model
---------------------
.. autoclass:: EfficientNet
   :members:
   :noindex:

Multimodal Concatenation
---------------------
.. autoclass:: MultimodalConcatenate
   :members:
   :noindex:


File Splitting AI
=====================

`[source] <https://github.com/konfuzio-ai/konfuzio-sdk/blob/master/konfuzio_sdk/trainer/file_splitting.py>`__

.. automodule:: konfuzio_sdk.trainer.file_splitting

Abstract File Splitting Model
---------------------
.. autoclass:: AbstractFileSplittingModel
   :members:
   :noindex:

Context Aware File Splitting Model
---------------------
.. autoclass:: ContextAwareFileSplittingModel
   :members:
   :noindex:

Multimodal File Splitting Model
---------------------
.. autoclass:: MultimodalFileSplittingModel
   :members:
   :noindex:

Splitting AI
---------------------
.. autoclass:: SplittingAI
   :members:
   :noindex:

AI Evaluation
=====================

`[source] <https://github.com/konfuzio-ai/document-ai-python-sdk/blob/master/konfuzio_sdk/evaluate.py>`__

Extraction AI Evaluation
---------------------
.. autoclass:: konfuzio_sdk.evaluate.ExtractionEvaluation
   :members:
   :noindex:

Categorization AI Evaluation
---------------------
.. autoclass:: konfuzio_sdk.evaluate.CategorizationEvaluation
   :members:
   :noindex:

File Splitting AI Evaluation
---------------------
.. autoclass:: konfuzio_sdk.evaluate.FileSplittingEvaluation
   :members:
   :noindex:

