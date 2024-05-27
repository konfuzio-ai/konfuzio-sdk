Tutorials
===========================

Welcome to the Developer's Guide Tutorials section, where you'll find a comprehensive set of tutorials to help you make the most of our powerful AI tools. These tutorials are designed to guide you through various aspects of Document processing, from data preparation to advanced techniques like Named Entity Recognition (NER) and barcode scanning.

Getting Started
---------------

1. :doc:`Example usage of main Konfuzio concepts <tutorials/example-usage/index>`

Get to know how to operate main Konfuzio concepts like Documents and Project, as well as learn structure of a Project folder.

2. :doc:`Data Preparation <tutorials/data-preparation/index>`

Learn how to efficiently prepare your data for optimal processing. This tutorial covers data organization, cleaning, and formatting to ensure seamless integration with our AI models.

3. :doc:`Create, change and delete Annotations via API <tutorials/create-annotations-api/index>`

Get to know how to create, change and delete different types of Annotations using methods from `konfuzio_sdk.api`.

.. toctree::
   :maxdepth: 1
   tutorials/example-usage/index
   tutorials/data-preparation/index
   tutorials/create-annotations-api/index

Document Processing Essentials
------------------------------

.. toctree::
   :maxdepth: 1
   tutorials/set-category-manually/index
   tutorials/document_categorization/index
   tutorials/create-custom-categorization-ai/index
   tutorials/build-context-aware-file-splitting-model/index
   tutorials/context-aware-file-splitting-model/index
   tutorials/create-custom-splitting-ai/index
   tutorials/file-splitting-evaluation/index.md
   tutorials/tokenizers/index
   tutorials/information_extraction/index
   tutorials/upload-your-ai/index

4. :doc:`Categorize a Document manually <tutorials/set-category-manually/index>`

Get to know how to assign and change Category of Documents and its Pages manually.

5. :doc:`Categorize a Document using Categorization AI <tutorials/document_categorization/index>`

Master the art of categorizing Documents within your projects automatically. This tutorial provides step-by-step guidance on labeling and organizing Documents based on their content.

6. :doc:`Create your own Categorization AI <tutorials/create-custom-categorization-ai/index>`

Build your own Categorization AI and define a custom architecture or reuse any external model. This tutorial provides
guidance about constructing a class for the model for Document Categorization that can later be reused on Konfuzio app
or in an on-prem installation.

7. :doc:`Build a Context-Aware File Splitting Model <tutorials/build-context-aware-file-splitting-model/index>`

Learn how to build a lightweight model for splitting a multi-Document stream of Pages. This tutorial is a step-by-step guide
about the existing class of Konfuzio SDK.

8. :doc:`Train and use a Context-Aware File Splitting Model <tutorials/context-aware-file-splitting-model/index>`

Familiarize yourself with a simple fallback logic for splitting stream of Pages into separate sub-Documents.

9. :doc:`Create your own File Splitting AI <tutorials/create-custom-splitting-ai/index>`

Build a custom File Splitting AI and define your own architecture or reuse any external model. This tutorial provides
guidance about constructing a class for the model for File Splitting that can later be reused on Konfuzio app
or in an on-prem installation.

10. :doc:`Evaluate the performance of a File Splitting AI <tutorials/file-splitting-evaluation/index>`

Get to know how to work with FileSplittingEvaluation class, assess the performance of Splitting AIs and interpret the
results.

11. :doc:`Tokenization <tutorials/tokenizers/index>`

Delve into the world of Document tokenization, a crucial step in text analysis. This tutorial explores various tokenization techniques and their applications.

12. :doc:`Information Extraction <tutorials/information_extraction/index>`

Unlock the potential of extracting valuable information from unstructured text. This tutorial guides you through the process of identifying and labeling key details.

13. :doc:`Upload your AI model to use on Konfuzio app or an on-prem installation <tutorials/upload-your-ai/index>`

Learn how to proceed with your model after you built and trained it and upload it to use in production using API.


Advanced Techniques
-------------------

14. :doc:`Named Entity Recognition <tutorials/ner-ontonotes-fast/index>`

Take your text analysis to the next level with fast and accurate Named Entity Recognition using OntoNotes. This tutorial provides in-depth insights into NER techniques.

15. :doc:`Annual Reports Analysis <tutorials/annual-reports/index>`

Learn how to extract critical insights from annual reports using our advanced AI models. This tutorial is ideal for financial analysts and researchers.

Specialized Applications
------------------------

16. :doc:`Barcode Scanner <tutorials/barcode-scanner/index>`

Explore the capabilities of our barcode scanning tool. This tutorial demonstrates how to effortlessly extract information from barcodes in Documents.

17. :doc:`PDF Form Generator <tutorials/pdf-form-generator/index>`

Learn how to dynamically generate PDF forms using our AI-powered tools. This tutorial is perfect for streamlining Document creation processes.

18. :doc:`Regex-based Annotations <tutorials/regex_based_annotations/index>`

Harness the power of regular expressions for precise Document Annotations. This tutorial guides you through the process of using regex patterns effectively.

19. :doc:`Object Detection <tutorials/object-detection/index>`

Dive into object detection to detect document structures. This tutorial provides a step by step guide on how to train an object detection model on document structures.


Streamlined Operations
---------------------

20. :doc:`Data Validation <tutorials/data_validation/index>`

Ensure the accuracy and integrity of your data with effective validation techniques. This tutorial provides best practices for maintaining high-quality data sets.

21. :doc:`Outlier Annotations <tutorials/outlier-annotations/index>`

Discover how to identify and handle outliers in your Document processing pipeline. This tutorial offers strategies for accurate Annotations.

22. :doc:`Async Upload with Callback <tutorials/async_upload_with_callback/index>`

Optimize your Document processing workflow with asynchronous upload and callback functionality. This tutorial enhances the efficiency of large-scale operations.

Dive into these tutorials and elevate your Document processing capabilities. Whether you're a beginner or an experienced developer, you'll find valuable insights and practical techniques to enhance your projects. Happy coding!


.. toctree::
   :maxdepth: 1
   tutorials/ner-ontonotes-fast/index
   tutorials/annual-reports/index
   tutorials/barcode-scanner/index
   tutorials/pdf-form-generator/index
   tutorials/data_validation/index
   tutorials/outlier-annotations/index
   tutorials/regex_based_annotations/index
   tutorials/async_upload_with_callback/index
   tutorials/object-detection/index