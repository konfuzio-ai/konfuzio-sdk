## Document Categorization

### Working with the Category of a Document and its individual Pages

You can initialize a Document with a Category, which will count as if a human manually revised it.

.. literalinclude:: /sdk/boilerplates/test_document_categorization.py
   :language: python
   :lines: 13,39-42,44

If a Document is initialized with no Category, it can be manually set later.

.. literalinclude:: /sdk/boilerplates/test_document_categorization.py
   :language: python
   :lines: 46,48-54


If you use a Categorization AI to automatically assign a Category to a Document (such as the 
[NameBasedCategorizationAI](tutorials.html#name-based-categorization-ai)), each Page will be assigned a 
Category Annotation with predicted confidence information, and the following properties will be accessible. You can 
also find these documented under [API Reference - Document](sourcecode.html#document), 
[API Reference - Page](sourcecode.html#page) and 
[API Reference - Category Annotation](sourcecode.html#category-annotation).

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

### Name-based Categorization AI

Use the name of the Category as an effective fallback logic to categorize Documents when no Categorization AI is available:

.. literalinclude:: /sdk/boilerplates/test_document_categorization.py
   :language: python
   :lines: 2-3,11-13,15-30,32-37

### Model-based Categorization AI

Build, train and test a Categorization AI using Image Models and Text Models to classify the image and text of each Page.

For a list of available Models see [Available Categorization Models](#id2).

.. literalinclude:: /sdk/boilerplates/test_document_categorization.py
   :language: python
   :lines: 2,4,6,5,11-13,57-72,76,80-83,85-87,88-89,91

#### Available Categorization Models

When using `build_categorization_ai_pipeline`, you can select which Image Module and/or Text Module to use for 
classification. At least one betweem the Image Model or the Text Model must be specified. Both can also be used 
at the same time.

The list of available Categorization Models is implemented as an Enum containing the following elements:

.. literalinclude:: /sdk/boilerplates/test_document_categorization.py
   :language: python
   :lines: 6,95-115

See more details about these Categorization Models under [API Reference - Categorization AI](sourcecode.html#categorization-ai).

### Categorization AI Overview Diagram

In the first diagram, we show the class hierarchy of the available Categorization Models within the SDK. Note that the 
Multimodal Model simply consists of a Multi Layer Perceptron to concatenate the feature outputs of a Text Model and an 
Image Model, such that the predictions from both Models can be unified in a unique Category prediction.

In the second diagram, we show how these models are contained within a Model-based Categorization AI. The 
[Categorization AI](https://dev.konfuzio.com/sdk/sourcecode.html#categorization-ai) class provides the high level 
interface to categorize Documents, as exemplified in the code examples above. It uses a Page Categorization Model 
to categorize each Page. The Page Categorization Model is a container for Categorization Models: it wraps the feature 
output layers of each contained Model with a Dropout Layer and a Fully Connected Layer.

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers tags lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;url&quot;:&quot;https://raw.githubusercontent.com/konfuzio-ai/konfuzio-sdk/master/docs/sdk/examples/document_categorization/CategorizationAI.drawio&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/embed2.js?&fetch=https%3A%2F%2Fraw.githubusercontent.com%2Fkonfuzio-ai%2Fkonfuzio-sdk%2Fmaster%2Fdocs%2Fsdk%2Fexamples%2Fdocument_categorization%2FCategorizationAI.drawio"></script>
