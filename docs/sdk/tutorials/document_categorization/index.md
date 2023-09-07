.. _document-categorization-tutorials:
## Document Categorization

<details>
<summary><b>Table of contents</b></summary>

[Setting a Category manually](#setting-a-category-manually)
- [Initialize a Document within a Category](#initialize-a-document-within-a-category)
- [Set a Document Category with `document.set_category` function](#set-a-document-category-with-document.set_category-function)
    - [Documents with no Category](#documents-with-no-category)
    - [Changing the Category](#changing-the-category)
    
[Assigning a Category with Categorization AI](#assigning-a-category-with-categorization-ai)
- [Name-based Categorization AI](#name-based-categorization-ai)
- [Model-based Categorization AI](#model-based-categorization-ai)
    - [Using model-based Categorization AI](#using-model-based-categorization-ai)
    - [Create custom Categorization AI](#create-custom-categorization-ai)
        - [Train custom Categorization AI locally and save it](#train-custom-categorization-ai-locally-and-save-it)
            - [Example usage of your Custom Categorization AI](#example-usage-of-your-custom-categorization-ai)
        - [Upload custom Categorization AI to the Konfuzio Server. How to remove the uploaded model](#upload-custom-categorization-ai-to-konfuzio-server-how-to-remove-the-uploaded-model)
    - [Categorization AI Overview Diagrams](#categorizaion-ai-overview-diagrams)
        - [Categorization AI Class Hierarchy Diagram](#categorization-ai-class-hierarchy-diagram)
        - [Model-based Categorization AI Structure Diagram](#model-based-categorization-ai-structure-diagram)
</details>
 
When uploading a Document to Konfuzio, the first step is to **assign it to a Category**. You can do this in either way:
-	[Manually](#setting-category-manually)
-	Automatically [using  Categorization AI]().

### Setting a Category manually

There are two options to set a Category manually for a Document: 
- [Initialize a Document within a Category](#initialize-a-document-within-a-category)
OR
- [Set a Document’s Category with `my_document.set_category` function](#set-a-document-category-with-document.set_category-function)

#### Initialize a Document within a Category

The first option is to initialize a Document within a Category. For this we add the `category=my_category` parameter to the Document object that we are initializing.

project = Project(id_=YOUR_PROJECT_ID)
my_category = project.get_category_by_id(YOUR_CATEGORY_ID)

my_document = Document(text="My text.", project=project, category=my_category)
assert my_document.category == my_category
my_document.set_category(my_category)
assert my_document.category_is_revised is True

```python
project = Project(id_=YOUR_PROJECT_ID)
my_category = project.get_category_by_id(YOUR_CATEGORY_ID)

my_document = Document(text="My text.", project=project, category=my_category)
assert my_document.category == my_category
```

#### Set a Document Category with `document.set_category` function

The second option is to set a Document’s Category in either of the cases:
- after the document has been initialized **with no category**
- in case we want to **re-assign a new Category** to the existing Document object.

We use `document.set_category` function for this.

```python
project = Project(id_=YOUR_PROJECT_ID)
my_category = project.get_category_by_id(YOUR_CATEGORY_ID)

my_document = Document(text="My text.", project=project)

my_document.set_category(my_category)
assert my_document.category_is_revised is True
```

##### Documents with no Category

If a Document is initialized with **no Category** (like in the code example above), it will automatically get the `category=no_category` parameter.

A Category can be **removed** from a Document by using `document.set_category(None)` .

```python
document.set_category(None)
assert document.category == project.no_category
```

##### Changing the Category

To assign a **different Category** to the same Document we again use the `document.set_category()` function.

```python
document.set_category(my_category)
assert document.category == my_category
assert document.category_is_revised is True 
```

> Note: Either of the options will set the Category **for the Document** and  **for all Pages in a Document**. To test this, please use the code below.

```python
for page in document.pages():
    assert page.category == my_category
```

### Assigning a Category with Categorization AI
You can assign a Category to a Document automatically with the help of the **Categorization AI**.  
The Categorization AI will generate predicted values and use them to create a **Category Annotation** with the following values:
1. Predicted value for the `.category` parameter of a Document,
2. Predicted **confidence information**.

Here is the **list of all properties** that can be used on Category, Page or Document level.

| Property                                          | Description                                                                                                                                                                                                           | More Info                                                  |
|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| `CategoryAnnotation.category`                     | The AI predicted Category of this Category Annotation.                                                                                                                                                                |[API Reference - Document](../../sourcecode.html#document) |
| `CategoryAnnotation.confidence`                   | The AI predicted confidence of this Category Annotation.                                                                                                                                                              | [API Reference - Document](../../sourcecode.html#document) |
| `Document.category_annotations`                   | List of predicted Category Annotations at the Document level.                                                                                                                                                         | [API Reference - Document](../../sourcecode.html#document) |
| `Document.maximum_confidence_category_annotation` | Get the maximum confidence predicted Category Annotation, or the human revised one if present.                                                                                                                        | [API Reference - Document](../../sourcecode.html#document) |
| `Document.maximum_confidence_category`            | Get the maximum confidence predicted Category or the human revised one if present.                                                                                                                                    | [API Reference - Document](../../sourcecode.html#document) |
| `Document.category`                               | Returns a Category only if all Pages have same Category, otherwise None. In that case, it hints to the fact that the Document should probably be revised or split into Documents with consistently categorized Pages. | [API Reference - Document](../../sourcecode.html#document) |
| `Page.category_annotations`                       | List of predicted Category Annotations at the Page level.                                                                                                                                                             | [API Reference - Page](../../sourcecode.html#page)         |
| `Page.maximum_confidence_category_annotation`     | Get the maximum confidence predicted Category Annotation or the one revised by the user for this Page.                                                                                                                | [API Reference - Page](../../sourcecode.html#page)         |
| `Page.category`                                   | Get the maximum confidence predicted Category or the one revised by user for this Page.                                                                                                                               | [API Reference - Page](../../sourcecode.html#page)         |


To categorize a Document with a Categorization AI, we have two main options:
- [Name-based Categorization AI](name-based-categorization-ai) (when no model-based Categorization AI is available)
- [Model-based Categorization AI](model-based-categorization-ai) (better results, but more complex)


#### Name-based Categorization AI
The name-based Categorization AI uses the **name of the Category** to categorize Documents. We can use it as a fallback logic when no no model-based Categorization AI is available.

1. Set up your Project.

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.document_categorization import NameBasedCategorizationAI

project = Project(id_=YOUR_PROJECT_ID)
```

2. Initialize the Categorization Model.

```python
categorization_model = NameBasedCategorizationAI(project.categories)
```

3. Retrieve a Document to categorize.
```python
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)
```

4. The Categorization Model returns a copy of the SDK Document with the Category attribute (use `inplace=True` to maintain the original Document instead). 
If the input Document is already categorized, the existing Category is used.
```python
result_doc = categorization_model.categorize(document=test_document)
```
> Note: Use `recategorize=True` if you want to force re-categorization.

5. Each Page is categorized individually.

 ```python
for page in result_doc.pages():
    assert page.category == project.categories[0]
    print(f"Found category {page.category} for {page}")
```
6. The Category of the Document is defined when Categories for all Pages are equal.
If the Document contains mixed Categories, only the Page level Category will be defined, and the Document level Category will be `NO_CATEGORY`.
```python
print(f"Found category {result_doc.category} for {result_doc}")
```

#### Model-based Categorization AI

Model-based Categorization AI is a preferable option to get **better results**, in comparison to the Name-based Categorization AI. 

Follow the steps below to **build, train and test Categorization AI** using Image Models and Text Models to classify the image and text of each Page.
> Note: To prepare the data for training and testing your AI, follow the [Data Preparation Tutorial](https://dev.konfuzio.com/sdk/tutorials/data-preparation/index.html#prepare-the-data-for-training-and-testing-the-ai).

##### Using model-based Categorization AI

1.  Import the data and the models.
```python
from konfuzio_sdk.data import Project, Document
from konfuzio_sdk.trainer.document_categorization import build_categorization_ai_pipeline
from konfuzio_sdk.trainer.document_categorization import ImageModel, TextModel, CategorizationAI
```


2. Set up your Project.
```python
project = Project(id_=YOUR_PROJECT_ID)
```
3. Build the Categorization AI architecture. Use the templates
of the pre-built Image and Text classification Models:

```python
categorization_pipeline = build_categorization_ai_pipeline(
    categories=project.categories,
    documents=project.documents,
    test_documents=project.test_documents,
    image_model=ImageModel.EfficientNetB0,
    text_model=TextModel.NBOWSelfAttention,
)
```

Please refer to the [Categorization AI Models](#categorization-ai-models) section to view all Categorization Models available.

4. Train the AI.

```python
categorization_pipeline.fit(n_epochs=1, optimizer={'name': 'Adam'})
```

5. Evaluate the AI.

```python
data_quality = categorization_pipeline.evaluate(use_training_docs=True)
ai_quality = categorization_pipeline.evaluate()
assert data_quality.f1(None) == 1.0
assert ai_quality.f1(None) == 1.0
```

6. Categorize a Document

```python
document = project.get_document_by_id(YOUR_DOCUMENT_ID)
categorization_result = categorization_pipeline.categorize(document=document)
assert isinstance(categorization_result, Document)
for page in categorization_result.pages():
    print(f"Found category {page.category} for {page}")
```

7. Save and load a pickle file for the AI

```python
pickle_ai_path = categorization_pipeline.save()
categorization_pipeline = CategorizationAI.load_model(pickle_ai_path)
```

In the following section all available [Categorization Models](categorization-ai-models) are listed.

##### Categorization AI Models
When building the Categorization AI architecture at Step 3[link] above we use  `build_categorization_ai_pipeline`. Here you can choose which **Image Model** and/or **Text Model** to use for classification. You have the following options:
-	use a chosen Image Model
-	use a chosen Text Model
-	use both - a chosen Image Model and a chosen Text Model

> Note: we should specify at least one Model here (Image or Text). Both an Image and a Text Models can also be used at the same time.

**The list of available Categorization Models** is implemented as an `Enum` containing the following elements:

```python
from konfuzio_sdk.trainer.document_categorization import ImageModel, TextModel

# Image Models
ImageModel.VGG11
ImageModel.VGG13
ImageModel.VGG16
ImageModel.VGG19
ImageModel.EfficientNetB0
ImageModel.EfficientNetB1
ImageModel.EfficientNetB2
ImageModel.EfficientNetB3
ImageModel.EfficientNetB4
ImageModel.EfficientNetB5
ImageModel.EfficientNetB6
ImageModel.EfficientNetB7
ImageModel.EfficientNetB8

# Text Models
TextModel.NBOW
TextModel.NBOWSelfAttention
TextModel.LSTM
TextModel.BERT
```

For more details Categorization about these Categorization Models refer to [API Reference - Categorization AI](../../sourcecode.html#categorization-ai).

#### Create custom Categorization AI

This section explains, how to:
1. [**Train** custom Categorization AI locally and **save** it](train-custom-categorization-ai-locally-and-save-it)
2. [**Upload** it to the Konfuzio Server](). 

> Note 1: If you run this tutorial in **Colab** and experience any version compatibility issues when working with the SDK, **restart the runtime and initialize the SDK once again**; this will resolve the issue.

> Note 2: If you **already have some document-processing architecture**, you don’t necessarily need to create the AI from scratch. Just **wrap it into the class** that corresponds to our Categorization AI structure. To learn about requirements for this, follow the steps below.

> Note 3: Currently, the Server supports AI models created using `torch<2.0.0`.

##### Train custom Categorization AI locally and save it

By default, any Categorization AI class should derive from the `AbstractCategorizationModel` class and implement the following methods:
- `fit`
- `_categorize_page`
- `save`

1.  Import the data
```python
from konfuzio_sdk.trainer.document_categorization import AbstractCategorizationAI
from konfuzio_sdk.data import Page, Category
from typing import List
```

2. Create a custom CategorizationAI class according to this template:

```python
class CustomCategorizationAI(AbstractCategorizationAI):
    def __init__(self, categories: List[Category], *args, **kwargs):
        # a list of Categories the AI will choose from
        super().__init__(categories)
        pass
```

 
3. Set the `fit` method according to this template:

```python
def fit(self):
        pass
```

  In the `fit` method we define architecture and training applied to the model, for example, a NN architecture or a custom hardcoded logic.
 For example, you can define it in this way:

```python
self.classifier_iterator = build_document_classifier_iterator(
         self.documents,
        self.train_transforms,
        use_image = True,
        use_text = False,
        device='cpu',
         )
self.classifier._fit_classifier(self.classifier_iterator, **kwargs)
```
>  Note 1: This method **does not return anythin**g.
However, if you provide the `self.model`  attribute, it will **modify** it.

> Note 2: This method may be implemented as a **no-op** , also if you **provide the trained model in other ways**.

3. Define the `_categorize_page` method according to this template:

```python
def _categorize_page(self, page: Page) -> Page:
        pass
```

In the `_categorize_page` we define how the model assigns a Category to a Page.
For example, you can define it in this way:
```python
     predicted_category_id, predicted_confidence = self._predict(page_image)
    
     for category in self.categories:
         if category.id_ == predicted_category_id:
            _ = CategoryAnnotation(category=category, confidence=predicted_confidence, page=page)
```
> Note: The result of the extraction should be the **input Page** with the added **Categorization attribute** `Page.category`.

4. Define the `save` method according to this template:

```python
def save(self, path: str):
        pass
```

In `save` we define how to save a model in a `.pt` format.
For example, we can do this in the same way it is defined in the CategorizationAI:
```python
    data_to_save = {
            'tokenizer': self.tokenizer,
            'image_preprocessing': self.image_preprocessing,
            'image_augmentation': self.image_augmentation,
            'text_vocab': self.text_vocab,
            'category_vocab': self.category_vocab,
            'classifier': self.classifier,
            'eval_transforms': self.eval_transforms,
            'train_transforms': self.train_transforms,
            'model_type': 'CategorizationAI',
             }
torch.save(data_to_save, path)
```

###### Example usage of your Custom Categorization AI

After you have created your custom Categorization AI, let’s follow the steps of its usage example.

1. Import the libraries and the data.

```python
import os
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.document_categorization import (
    CategorizationAI,
    EfficientNet,
    PageImageCategorizationModel,
)
```

2. Initialize Project and provide AI training and test data
See https://dev.konfuzio.com/sdk/get_started.html#example-usage

```python
project = Project(id_=YOUR_PROJECT_ID)

categorization_pipeline = CategorizationAI(project.categories)
categorization_pipeline.categories = project.categories
categorization_pipeline.documents = [
    document for category in categorization_pipeline.categories for document in category.documents()
]
categorization_pipeline.test_documents = [
    document for category in categorization_pipeline.categories for document in category.test_documents()
]
```

3. Initialize all necessary parts of the AI – in this example we run an AI that **uses images** and does not use text.

```python
categorization_pipeline.category_vocab = categorization_pipeline.build_template_category_vocab()

# image processing model
image_model = EfficientNet(name='efficientnet_b0')

# building a classifier for the page images
categorization_pipeline.classifier = PageImageCategorizationModel(
    image_model=image_model,
    output_dim=len(categorization_pipeline.category_vocab),
)

categorization_pipeline.build_preprocessing_pipeline(use_image=True)
```

4. Fit the AI

```python
categorization_pipeline.fit(n_epochs=1, optimizer={'name': 'Adam'})
```

5. Evaluate the AI

```python
data_quality = categorization_pipeline.evaluate(use_training_docs=True)
ai_quality = categorization_pipeline.evaluate(use_training_docs=False)
```


6. Categorize a Document

```python
document = project.get_document_by_id(YOUR_DOCUMENT_ID)
categorization_result = categorization_pipeline.categorize(document=document)
for page in categorization_result.pages():
    print(f"Found category {page.category} for {page}")
print(f"Found category {categorization_result.category} for {categorization_result}")
```

7. Save and load a pickle file for the model

```python
pickle_model_path = categorization_pipeline.save(reduce_weight=False)
categorization_pipeline_loaded = CategorizationAI.load_model(pickle_model_path)
```


#### Upload custom Categorization AI to the Konfuzio Server. How to remove the uploaded model
After you have trained and saved your custom AI, you can **upload** it:
- Using the steps from the [tutorial](https://help.konfuzio.com/tutorials/migrate-trained-ai-to-an-new-project-to-annotate-documents-faster/index.html#upload-extraction-or-category-ai-to-target-instance)
OR
- Using the `upload_ai_model()` method, provided that you have Superuser rights.

To **remove** the uploaded model use the `delete_ai_model()` method.
```python
from konfuzio_sdk.api import upload_ai_model, delete_ai_model

#Upload the saved model to the server
model_id = upload_ai_model(pickle_model_path)

# Remove the model
delete_ai_model(model_id, ai_type='categorization')
```

### Categorization AI Overview Diagrams

#### Categorization AI Class Hierarchy Diagram

In the first diagram we show **class hierarchy** of the available Categorization Models within the SDK.
> Note: the Multimodal Model simply consists of a **Multi Layer Perceptron** to *concatenate* the feature outputs of a Text Model and an Image Model, so that the predictions from both Models can be unified in a unique **Category prediction**.

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers tags lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;url&quot;:&quot;https://raw.githubusercontent.com/konfuzio-ai/konfuzio-sdk/master/docs/sdk/tutorials/document_categorization/CategorizationAI.drawio&quot;}"></div>

#### Model-based Categorization AI Structure Diagram

In the second diagram, we show how Categorization Models are contained within a Model-based Categorization AI.

The **Categorization AI** class provides the **high level interface** to categorize Documents, as exemplified in the code examples above. It uses a Page Categorization Model to categorize each Page.

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers tags lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;url&quot;:&quot;https://raw.githubusercontent.com/konfuzio-ai/konfuzio-sdk/master/docs/sdk/tutorials/document_categorization/CategorizationAI.drawio&quot;}"></div>

The **Page Categorization Model** is a container for Categorization Models: it wraps the feature output layers of each contained Model with the Dropout Layer and the Fully Connected Layer.

<script type="text/javascript" src="https://viewer.diagrams.net/embed2.js?&fetch=https%3A%2F%2Fraw.githubusercontent.com%2Fkonfuzio-ai%2Fkonfuzio-sdk%2Fmaster%2Fdocs%2Fsdk%2Ftutorials%2Fdocument_categorization%2FCategorizationAI.drawio"></script>
