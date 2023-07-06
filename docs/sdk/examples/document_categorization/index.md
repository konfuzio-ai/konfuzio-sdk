## Document Categorization

### Working with the Category of a Document and its individual Pages

You can initialize a Document with a Category, which will count as if a human manually revised it.

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
   :start-after: start name-based
   :end-before: end name-based
   :dedent: 4

### Model-based Categorization AI

Build, train and test a Categorization AI using Image Models and Text Models to classify the image and text of each Page.

For a list of available Models see [Available Categorization Models](#categorization-models-available).

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

.. _categorization-models-available:
#### Available Categorization Models

When using `build_categorization_ai_pipeline`, you can select which Image Module and/or Text Module to use for 
classification. At least one between the Image Model or the Text Model must be specified. Both can also be used 
at the same time.

The list of available Categorization Models is implemented as an Enum containing the following elements:

See more details about these Categorization Models under [API Reference - Categorization AI](sourcecode.html#categorization-ai).

### Customize Categorization AI

This section explains how to train a custom Categorization AI locally, how to save it and upload it to the Konfuzio 
Server. 

By default, any [Categorization AI](sourcecode.html#categorization-ai) class should derive from the 
`AbstractCategorizationModel` class and implement the following interface:

.. exec_code::

   # --- hide: start ---
   from konfuzio_sdk.trainer.document_categorization import AbstractCategorizationAI
   from konfuzio_sdk.data import Page, Category
   from typing import List
   # --- hide: stop ---
   class CustomCategorizationAI(AbstractCategorizationAI):
       def __init__(self, categories: List[Category], *args, **kwargs):
           super().__init__(categories)
           pass

       # initialize key variables required by the custom AI:
       # for instance, self.documents and self.test_documents to train and test the AI on, self.categories to determine
       # which Categories will the AI be able to predict

       def fit(self):
          pass

       # Define architecture and training that the model undergoes, i.e. a NN architecture or a custom hardcoded logic
       # for instance:
       #
       # self.classifier_iterator = build_document_classifier_iterator(
       #             self.documents,
       #             self.train_transforms,
       #             use_image = True,
       #             use_text = False,
       #             device='cpu',
       #         )
       # self.classifier._fit_classifier(self.classifier_iterator, **kwargs)
       #
       # This method is allowed to be implemented as a no-op if you provide the trained model in other ways

       def _categorize_page(self, page: Page) -> Page:
           pass

       # define how the model assigns a Category to a Page.
       # for instance:
       #
       # predicted_category_id, predicted_confidence = self._predict(page_image)
       #
       # for category in self.categories:
       #     if category.id_ == predicted_category_id:
       #         _ = CategoryAnnotation(category=category, confidence=predicted_confidence, page=page)
       #
       # **NB:** The result of extraction must be the input Page with added Categorization attribute `Page.category`

       def save(self, path: str):
          pass

       # define how to save a model in a .pt format – for example, in a way it's defined in the CategorizationAI
       #
       #  data_to_save = {
       #             'tokenizer': self.tokenizer,
       #             'image_preprocessing': self.image_preprocessing,
       #             'image_augmentation': self.image_augmentation,
       #             'text_vocab': self.text_vocab,
       #             'category_vocab': self.category_vocab,
       #             'classifier': self.classifier,
       #             'eval_transforms': self.eval_transforms,
       #             'train_transforms': self.train_transforms,
       #             'model_type': 'CategorizationAI',
       #         }
       # torch.save(data_to_save, path)

Example usage of your Custom Categorization AI:

.. exec_code::
   
   # --- hide: start ---
   from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID
   YOUR_PROJECT_ID = TEST_PROJECT_ID
   YOUR_DOCUMENT_ID = TEST_DOCUMENT_ID
   # --- hide: stop ---
   import os
   from konfuzio_sdk.data import Project
   from konfuzio_sdk.trainer.document_categorization import (
        CategorizationAI,
        EfficientNet,
        PageImageCategorizationModel,
    )

   # Initialize Project and provide the AI training and test data
   project = Project(id_=YOUR_PROJECT_ID)  # see https://dev.konfuzio.com/sdk/get_started.html#example-usage

   categorization_pipeline = CategorizationAI(project.categories)
   categorization_pipeline.categories = project.categories
   categorization_pipeline.documents = [
        document for category in categorization_pipeline.categories for document in category.documents()
    ]
   categorization_pipeline.test_documents = [
        document for category in categorization_pipeline.categories for document in category.test_documents()
    ]
   # --- hide: start ---
   categorization_pipeline.documents = categorization_pipeline.documents[:5]
   categorization_pipeline.test_documents = categorization_pipeline.test_documents[:5]
   # --- hide: stop ---
   # initialize all necessary parts of the AI – in the example we run an AI that uses images and does not use text
   categorization_pipeline.category_vocab = categorization_pipeline.build_template_category_vocab()
   # image processing model
   image_model = EfficientNet(name='efficientnet_b0')
   # building a classifier for the page images
   categorization_pipeline.classifier = PageImageCategorizationModel(
        image_model=image_model,
        output_dim=len(categorization_pipeline.category_vocab),
    )
   categorization_pipeline.build_preprocessing_pipeline(use_image=True)
   # fit the AI
   categorization_pipeline.fit(n_epochs=1, optimizer={'name': 'Adam'})

   # evaluate the AI
   data_quality = categorization_pipeline.evaluate(use_training_docs=True)
   ai_quality = categorization_pipeline.evaluate(use_training_docs=False)

   # Categorize a Document
   document = project.get_document_by_id(YOUR_DOCUMENT_ID)
   categorization_result = categorization_pipeline.categorize(document=document)
   # --- skip: start ---
   for page in categorization_result.pages():
      print(f"Found category {page.category} for {page}")
   print(f"Found category {categorization_result.category} for {categorization_result}")
   # --- skip: stop ---
   # Save and load a pickle file for the model
   pickle_model_path = categorization_pipeline.save(reduce_weight=False)
   categorization_pipeline_loaded = CategorizationAI.load_model(pickle_model_path)
   # --- hide: start ---
   os.remove(pickle_model_path)
   assert 63 in data_quality.category_ids
   assert 63 in ai_quality.category_ids
   assert isinstance(categorization_pipeline_loaded, CategorizationAI)
   

After you have trained your custom AI, you can upload it using the steps from the [tutorial](https://help.konfuzio.com/tutorials/migrate-trained-ai-to-an-new-project-to-annotate-documents-faster/index.html#upload-extraction-or-category-ai-to-target-instance)
or using the method `upload_ai_model()`. You can also remove an uploaded model by using `delete_ai_model()`.

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

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers tags lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;url&quot;:&quot;https://raw.githubusercontent.com/konfuzio-ai/konfuzio-sdk/master/docs/sdk/examples/document_categorization/CategorizationAI.drawio&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/embed2.js?&fetch=https%3A%2F%2Fraw.githubusercontent.com%2Fkonfuzio-ai%2Fkonfuzio-sdk%2Fmaster%2Fdocs%2Fsdk%2Fexamples%2Fdocument_categorization%2FCategorizationAI.drawio"></script>
