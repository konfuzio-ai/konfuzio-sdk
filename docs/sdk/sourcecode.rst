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

API call wrappers
=====================

`[source] <https://github.com/konfuzio-ai/konfuzio-sdk/blob/master/konfuzio_sdk/api.py>`__

.. automodule:: konfuzio_sdk.api

TimeoutHTTPAdapter
---------------------
.. autoclass:: TimeoutHTTPAdapter
   :members:
   :noindex:

.. autofunction:: init_env
.. autofunction:: konfuzio_session
.. autofunction:: get_project_list
.. autofunction:: get_project_details
.. autofunction:: get_project_labels
.. autofunction:: get_project_label_sets
.. autofunction:: create_new_project
.. autofunction:: get_document_details
.. autofunction:: get_document_annotations
.. autofunction:: get_document_bbox
.. autofunction:: get_page_image
.. autofunction:: post_document_annotation
.. autofunction:: change_document_annotation
.. autofunction:: delete_document_annotation
.. autofunction:: update_document_konfuzio_api
.. autofunction:: download_file_konfuzio_api
.. autofunction:: get_results_from_segmentation
.. autofunction:: get_project_categories
.. autofunction:: upload_ai_model
.. autofunction:: delete_ai_model
.. autofunction:: update_ai_model
.. autofunction:: get_all_project_ais
.. autofunction:: export_ai_models


CLI tools
=====================

`[source] <https://github.com/konfuzio-ai/konfuzio-sdk/blob/master/konfuzio_sdk/cli.py>`__

.. automodule:: konfuzio_sdk.cli

.. autofunction:: parse_args
.. autofunction:: credentials

Extras
=====================

`[source] <https://github.com/konfuzio-ai/konfuzio-sdk/blob/master/konfuzio_sdk/extras.py>`__

.. automodule:: konfuzio_sdk.extras

PackageWrapper
---------------------
.. autoclass:: PackageWrapper
   :members:
   :noindex:

ModuleWrapper
---------------------
.. autoclass:: ModuleWrapper
   :members:
   :noindex:

Normalization
=====================

`[source] <https://github.com/konfuzio-ai/konfuzio-sdk/blob/master/konfuzio_sdk/normalize.py>`__

.. automodule:: konfuzio_sdk.normalize

.. autofunction:: normalize_to_float
.. autofunction:: normalize_to_positive_float
.. autofunction:: normalize_to_percentage
.. autofunction:: normalize_to_date
.. autofunction:: normalize_to_bool
.. autofunction:: roman_to_float
.. autofunction:: normalize

Utils
=====================

`[source] <https://github.com/konfuzio-ai/konfuzio-sdk/blob/master/konfuzio_sdk/utils.py>`__

.. automodule:: konfuzio_sdk.utils

.. autofunction:: sdk_isinstance
.. autofunction:: exception_or_log_error
.. autofunction:: get_id
.. autofunction:: is_file
.. autofunction:: memory_size_of
.. autofunction:: normalize_memory
.. autofunction:: get_timestamp
.. autofunction:: load_image
.. autofunction:: get_file_type
.. autofunction:: get_file_type_and_extension
.. autofunction:: does_not_raise
.. autofunction:: convert_to_bio_scheme
.. autofunction:: slugify
.. autofunction:: amend_file_name
.. autofunction:: amend_file_path
.. autofunction:: get_sentences
.. autofunction:: map_offsets
.. autofunction:: detectron_get_paragraph_bboxes
.. autofunction:: iter_before_and_after
.. autofunction:: get_sdk_version
.. autofunction:: get_spans_from_bbox
.. autofunction:: normalize_name

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

Textual File Splitting Model
---------------------
.. autoclass:: TextualFileSplittingModel
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

Evaluation Calculator
---------------------
.. autoclass:: konfuzio_sdk.evaluate.EvaluationCalculator
   :members:
   :noindex:

.. autofunction:: konfuzio_sdk.evaluate.grouped
.. autofunction:: konfuzio_sdk.evaluate.compare

Trainer utils
=====================

`[source] <https://github.com/konfuzio-ai/konfuzio-sdk/blob/master/konfuzio_sdk/trainer/utils.py>`__

.. automodule:: konfuzio_sdk.trainer.utils

LoggerCallback
---------------------
.. autoclass:: konfuzio_sdk.trainer.utils.LoggerCallback
   :members:
   :noindex:

BalancedLossTrainer
---------------------
.. autoclass:: konfuzio_sdk.trainer.utils.BalancedLossTrainer
   :members:
   :noindex: