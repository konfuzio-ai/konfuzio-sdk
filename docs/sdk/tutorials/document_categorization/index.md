.. _document-categorization-tutorials:
## Document Categorization

When uploading a Document to Konfuzio, the first step is to assign it to a :ref:`Category<category-concept>`. This 
can be done manually, or automatically using a Categorization AI.

### Setting the Category of a Document and its individual Pages Manually

You can initialize a Document with a :ref:`Category<category-concept>`. You can also use `Document.set_category` to set 
a Document's Category after it has been initialized. This will count as if a human manually revised it.

.. literalinclude:: /sdk/boilerplates/test_document_categorization.py
   :language: python
   :start-after: start init
   :end-before: end init
   :dedent: 4

If a Document is initialized with no Category, it will automatically be set to NO_CATEGORY. Another Category can be 
manually set later.

.. literalinclude:: /sdk/boilerplates/test_document_categorization.py
   :language: python
   :start-after: start no_category
   :end-before: end no_category
   :dedent: 4


If you use a Categorization AI to automatically assign a Category to a Document (such as the 
[NameBasedCategorizationAI](#name-based-categorization-ai), each Page will be assigned a 
Category Annotation with predicted confidence information, and the following properties will be accessible. You can 
also find these documented under [API Reference - Document](../../sourcecode.html#document), 
[API Reference - Page](../../sourcecode.html#page) and 
[API Reference - Category Annotation](../../sourcecode.html#category-annotation).

| Property                     | Description                                                                                                                                                                                                                       |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CategoryAnnotation.category`    | The AI predicted Category of this Category<br>Annotation.                                                                                                                                                                         |
| `CategoryAnnotation.confidence`  | The AI predicted confidence of this Category<br>Annotation.                                                                                                                                                                       |
| `Document.category_annotations`   | List of predicted Category Annotations at the<br>Document level.                                                                                                                                                                  |
| `Document.maximum_confidence_category_annotation`   | Get the maximum confidence predicted Category<br>Annotation, or the human revised one if present.                                                                                                                                 |
| `Document.maximum_confidence_category`   | Get the maximum confidence predicted Category<br>or the human revised one if present.                                                                                                                                             |
| `Document.category`  | Returns a Category only if all Pages have same<br>Category, otherwise None. In that case, it hints<br>to the fact that the Document should probably<br>be revised or split into Documents with<br>consistently categorized Pages. |
| `Page.category_annotations`   | List of predicted Category Annotations at the<br>Page level.                                                                                                                                                                      |
| `Page.maximum_confidence_category_annotation`   | Get the maximum confidence predicted Category<br>Annotation or the one revised by the user for this<br>Page.                                                                                                                      |
| `Page.category`  | Get the maximum confidence predicted Category<br>or the one revised by user for this Page.                                                                                                                                        |

To categorize a Document with a Categorization AI, we have two main options: the Name-based Categorization AI and the 
more complex Model-based Categorization AI.

### Name-based Categorization AI

The name-based Categorization AI is a good fallback logic using the name of the Category to categorize Documents when 
no model-based Categorization AI is available:

.. literalinclude:: /sdk/boilerplates/test_document_categorization.py
   :language: python
   :start-after: start name-based
   :end-before: end name-based
   :dedent: 4

### Model-based Categorization AI

For better results you can build, train and test a Categorization AI using Image Models and Text Models to classify 
the image and text of each Page:

.. literalinclude:: /sdk/boilerplates/test_document_categorization.py
   :language: python
   :start-after: start imports
   :end-before: end imports
   :dedent: 4
.. literalinclude:: /sdk/boilerplates/test_document_categorization.py
   :language: python
   :start-after: Start Build
   :end-before: End Build
   :dedent: 4

To prepare the data for training and testing your AI, follow the [data preparation tutorial](tutorials.html#tutorials.html#prepare-the-data-for-training-and-testing-the-ai).

For a list of available Models see all the available [Categorization Models](#categorization-ai-models) below.

#### Categorization AI Models

When using `build_categorization_ai_pipeline`, you can select which Image Module and/or Text Module to use for 
classification. At least one between the Image Model or the Text Model must be specified. Both can also be used 
at the same time.

The list of available Categorization Models is implemented as an Enum containing the following elements:

.. literalinclude:: /sdk/boilerplates/test_document_categorization.py
   :language: python
   :start-after: Start Models
   :end-before: End Models
   :dedent: 4

See more details about these Categorization Models under [API Reference - Categorization AI](../../sourcecode.html#categorization-ai).

### Create a custom Categorization AI

This section explains how to train a custom Categorization AI locally, how to save it and upload it to the Konfuzio 
Server. If you run this tutorial in Colab and experience any version compatibility issues when working with the SDK, restart the
runtime and initialize the SDK once again; this will resolve the issue.

Note: you don't necessarily need to create the AI from scratch if you already have some document-processing architecture.
You just need to wrap it into the class that corresponds to our Categorization AI structure. Follow the steps in this 
tutorial to find out what are the requirements for that.

Note: currently, the Server supports AI models created using `torch<2.0.0`.

By default, any [Categorization AI](../../sourcecode.html#categorization-ai) class should derive from the 
`AbstractCategorizationModel` class and implement the following methods:

.. literalinclude:: /sdk/boilerplates/test_custom_categorization_ai.py
   :language: python
   :start-after: start init
   :end-before: end init
   :dedent: 4

Example usage of your Custom Categorization AI:

.. literalinclude:: /sdk/boilerplates/test_custom_categorization_ai.py
   :language: python
   :start-after: start usage
   :end-before: end usage
   :dedent: 4
.. literalinclude:: /sdk/boilerplates/test_custom_categorization_ai.py
   :language: python
   :start-after: start fit
   :end-before: end fit
   :dedent: 4

After you have trained and saved your custom AI, you can upload it using the steps from the [tutorial](https://help.konfuzio.com/tutorials/migrate-trained-ai-to-an-new-project-to-annotate-documents-faster/index.html#upload-extraction-or-category-ai-to-target-instance)
or using the method `upload_ai_model()`, provided that you have the Superuser rights. You can also remove an uploaded 
model by using `delete_ai_model()`.

```python
from konfuzio_sdk.api import upload_ai_model, delete_ai_model

# upload a saved model to the server
model_id = upload_ai_model(pickle_model_path)

# remove model
delete_ai_model(model_id, ai_type='categorization')
```

### Categorization AI Overview Diagram

In the first diagram, we show the class hierarchy of the available Categorization Models within the SDK. Note that the 
Multimodal Model simply consists of a Multi Layer Perceptron to concatenate the feature outputs of a Text Model and an 
Image Model, such that the predictions from both Models can be unified in a unique Category prediction.

In the second diagram, we show how these models are contained within a Model-based Categorization AI. The 
[Categorization AI](https://dev.konfuzio.com/sdk/sourcecode.html#categorization-ai) class provides the high level 
interface to categorize Documents, as exemplified in the code examples above. It uses a Page Categorization Model 
to categorize each Page. The Page Categorization Model is a container for Categorization Models: it wraps the feature 
output layers of each contained Model with a Dropout Layer and a Fully Connected Layer.

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers tags lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;url&quot;:&quot;https://raw.githubusercontent.com/konfuzio-ai/konfuzio-sdk/master/docs/sdk/tutorials/document_categorization/CategorizationAI.drawio&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/embed2.js?&fetch=https%3A%2F%2Fraw.githubusercontent.com%2Fkonfuzio-ai%2Fkonfuzio-sdk%2Fmaster%2Fdocs%2Fsdk%2Ftutorials%2Fdocument_categorization%2FCategorizationAI.drawio"></script>
