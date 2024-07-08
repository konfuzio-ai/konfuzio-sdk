---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: konfuzio
    language: python
    name: python3
---

(info-extraction)=
## Information Extraction

---

**Prerequisites:**
- Familiarity with OOP principles
- Understanding of regular expressions.
- Understanding of evaluation measures for machine learning models.
- Data Layer of Konfuzio: Label, Annotation, Label Set, Span, Project, Document, Category
- AI Layer of Konfuzio: Information Extraction

**Difficulty:** Medium

**Goal:** Be able to build and deploy custom models for data extraction.

---

### Environment
You need to install the Konfuzio SDK before diving into the tutorial. \
To get up and running quickly, you can use our Colab Quick Start notebook. \
<a href="https://colab.research.google.com/github/konfuzio-ai/konfuzio-sdk/blob/master/notebooks/Quick_start_template_for_Konfuzio_SDK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

As an alternative you can follow the [installation section](../get_started.html#install-sdk) to install and initialize the Konfuzio SDK locally or on an environment of your choice.

### Introduction
Information Extraction is the process of obtaining information from the Document's unstructured text and assigning Labels to it. For example, Labels could be the Name, the Date, the Recipient, or any other field of interest in the Document.

Within Konfuzio, Documents are assigned a Category, which in turn can be associated to one or more Label Set(s) and therefore to a Label. To be precise, it is Label Set(s) that are associated to Categories, and not the other way around.

In this tutorial we will cover the following topics:
- How to train a custom Extraction AI that can be used with Konfuzio
- How to evaluate the performance of a trained Extraction AI model


### Train a custom date Extraction AI
This section explains how to create a custom Extraction AI locally, how to save it and upload it to the Konfuzio Server.

Any Extraction AI class should derive from the `AbstractExtractionAI` class and implement the `extract()` method. In this tutorial, we demonstrate how to create a simple custom Extraction AI that extracts dates provided in 
a certain format. Note that to enable Labels' and Label Sets' dynamic creation during extraction, you need to have Superuser rights and enable _dynamic creation_ in a [Superuser Project](https://help.konfuzio.com/modules/administration/superuserprojects/index.html#create-labels-and-label-sets).

We start by defining a custom class CustomExtractionAI that inherits from AbstractExtractionAI, containing a single method extract that takes a Document object as input and returns a modified Document.

Inside the `extract` method, the code first calls the parent method `super().extract()`. This method call retrieves a virtual Document with no Annotations and changes the Category to the one saved within the Extraction AI.

The code checks if a Label named 'Date' already exists in the Labels associated with the Category. It then either uses the existing Label, or creates a new one.

We use a regular expression (`r'(\d+/\d+/\d+)'`) to find matches for dates within the text of the Document. This regular expression looks for patterns of digits separated by forward slashes (e.g., 12/31/2023).

For each match found, it creates a Span object representing the start and end offsets of the matched text.

It then creates an Annotation object, associating it with the Document, then loops for each match found in the Document. Note that by default, only the Annotations with confidence higher than 10% will be shown in the extracted Document. This can be changed in the Label settings UI.

```python
import re

from konfuzio_sdk.data import Document, Span, Annotation, Label
from konfuzio_sdk.trainer.information_extraction import AbstractExtractionAI

class CustomExtractionAI(AbstractExtractionAI):
    def extract(self, document: Document) -> Document:
        document = super().extract(document)

        label_set = document.category.default_label_set

        label_name = 'Date'
        if label_name in [label.name for label in document.category.labels]:
            label = document.project.get_label_by_name(label_name)
        else:
            label = Label(text=label_name, project=document.project, label_sets=[label_set])
        annotation_set = document.default_annotation_set
        for re_match in re.finditer(r'(\d+/\d+/\d+)', document.text, flags=re.MULTILINE):
            span = Span(start_offset=re_match.span(1)[0], end_offset=re_match.span(1)[1])

            _ = Annotation(
                document=document,
                label=label,
                annotation_set=annotation_set,
                confidence=1.0,  
                spans=[span],
            )
        return document
```

We can now use this custom Extraction AI class. Let's start with making the necessary imports, initializing the Project, the Category and the AI:

```python tags=["remove-cell"]
# This is necessary to make sure we can import from 'tests'
import sys
sys.path.insert(0, '../../../../')

from tests.variables import TEST_PROJECT_ID, TEST_PAYSLIPS_CATEGORY_ID, TEST_DOCUMENT_ID
```

```python tags=["remove-output"]
import os
from konfuzio_sdk.data import Project

project = Project(id_=TEST_PROJECT_ID)
category = project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID)
categorization_pipeline = CustomExtractionAI(category)
```

Then, create a sample test Document to run the extraction on.
```python
example_text = """
    19/05/1996 is my birthday.
    04/07/1776 is the Independence day.
    """
sample_document = Document(project=project, text=example_text, category=category)
print(sample_document.text)
```

Run the extraction of a Document and print the extracted Annotations.
```python
extracted = categorization_pipeline.extract(sample_document)
for annotation in extracted.annotations(use_correct=False):
    print(annotation.offset_string)
```

Now we can save the AI and check that it is possible to load it afterwards.
```python
pickle_model_path = categorization_pipeline.save()
extraction_pipeline_loaded = CustomExtractionAI.load_model(pickle_model_path)
```

The custom Extraction AI we just prepared inherits from AbstractExtractionAI, which in turn inherits from [BaseModel](sourcecode.html#base-model). `BaseModel` provides `save` method that saves a model into a compressed pickle file that can be directly uploaded to the Konfuzio Server (see [Upload Extraction or Category AI to target instance](https://help.konfuzio.com/tutorials/migrate-trained-ai-to-an-new-project-to-annotate-documents-faster/index.html#upload-extraction-or-category-ai-to-target-instance)).

Activating the uploaded AI on the web interface will enable the custom pipeline on your self-hosted installation.

Note that if you want to create Labels and Label Sets dynamically (when running the AI, instead of adding them manually
on app), you need to enable creating them in the Superuser Project settings if you have the corresponding rights.

If you have the Superuser rights, it is also possible to upload the AI from your local machine using the 
`upload_ai_model()` as described in [Upload your AI](https://dev.konfuzio.com/sdk/tutorials/upload-your-ai/index.html).


### The Paragraph Custom Extraction AI
In [the Paragraph Tokenizer tutorial](https://dev.konfuzio.com/sdk/tutorials/tokenizers/index.html#paragraph-tokenization), we saw how we can use the Paragraph Tokenizer in `detectron` mode and with the `create_detectron_labels` option to segment a Document and create `figure`, `table`, `list`, `text` and `title` Annotations.

Here, we will see how we can use the Paragraph Tokenizer to create a Custom Extraction AI. We will create a simple wrapper around the Paragraph Tokenizer. This shows how you can create your own Custom Extraction AI which 
can be used in Konfuzio on-prem installations or in the [Konfuzio Marketplace](https://help.konfuzio.com/modules/marketplace/index.html).


We define a class that inherits from the Konfuzio `AbstractExtractionAI` class. This class provides the interface that we need to implement for our Custom Extraction AI. All Extraction AI models must inherit from this class.

We need to define what the model needs to be able to run. This will inform the Konfuzio Server what information needs to be made available to the model before running an extraction. If the model only needs text, you can add `requires_text = True` to make it explicit, but this is the default behavior. If the model requires Page images, you will need to add `requires_images = True`. Finally, in our case we also need to add `requires_segmentation = True` to inform the Server that the model needs the visual segmentation information created by the Paragraph Tokenizer in `detectron` mode.

We initialize the model by calling the `__init__` method of the parent class. The only required argument is the Category the Extraction AI will be used with. The Category is the Konfuzio object that contains all the Labels 
and Label Sets that the model will use to create Annotations. This means that you need to make sure that the Category object contains all the Labels and Label Sets that you need for your model. In our case, we need the `figure`, `table`, `list`, `text` and `title` Labels.

The `extract` method is the core of the Extraction AI. It takes a Document as input and returns a Document with Annotations. Make sure to do a `deepcopy` of the Document that is passed so that you add the new Annotations to a 
Virtual Document with no Annotations. The Annotations are created by the model and added to the Document. In our case, we simply call the Paragraph Tokenizer in `detectron` mode and with the `create_detectron_labels` option.

The `check_is_ready` method is used to check if the model is ready to be used. It should return `True` if the model is ready to extract, and `False` otherwise. Implementing this method is optional, but it is a good practice to make sure that the model is ready to be used.

```python
from konfuzio_sdk.trainer.information_extraction import AbstractExtractionAI
from konfuzio_sdk.tokenizer.paragraph_and_sentence import ParagraphTokenizer
from konfuzio_sdk.data import Category, Document, Project, Label

class ParagraphExtractionAI(AbstractExtractionAI):
    requires_images = True
    requires_text = True
    requires_segmentation = True

    def __init__(self, category: Category = None, *args, **kwargs,):
        """Initialize ParagraphExtractionAI."""
        super().__init__(category=category, *args, **kwargs)
        self.tokenizer = ParagraphTokenizer(mode='detectron', create_detectron_labels=True)    


    def extract(self, document: Document) -> Document:
        """
        Infer information from a given Document.
        """
        inference_document = super().extract(document)
        inference_document = self.tokenizer.tokenize(inference_document)

        return inference_document

    def check_is_ready(self):
        """
        Check if the ExtractionAI is ready for the inference.
        """
        super().check_is_ready()

        self.project.get_label_by_name('figure')
        self.project.get_label_by_name('table')
        self.project.get_label_by_name('list')
        self.project.get_label_by_name('text')
        self.project.get_label_by_name('title')

        return True        
```

Now that our custom Extraction AI is ready we can test it. First, we check that the category of interest indeed contains all Labels and create those that do not exist.

```python tags=["remove-output"]
project = Project(id_=TEST_PROJECT_ID)
category = project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID)

labels = ['figure', 'table', 'list', 'text', 'title']
label_set = project.get_label_set_by_name(category.name) 

for label_name in labels:
    try:
        project.get_label_by_name(label_name)
    except IndexError:
        Label(project=project, text=label_name, label_sets=[label_set])
```

We can now use our custom extraction model to extract data from a Document. 

```python tags=["remove-output"]
document = project.get_document_by_id(TEST_DOCUMENT_ID)
paragraph_extraction_ai = ParagraphExtractionAI(category=category)

assert paragraph_extraction_ai.check_is_ready() is True

extracted_document = paragraph_extraction_ai.extract(document)
```

Let's see all the created Annotations.
```python
print(extracted_document.annotations(use_correct=False))  
```

We then save the model as a pickle file, so that we can upload it to the Konfuzio Server:

```python tags=["remove-output"]
model_path = paragraph_extraction_ai.save()
```

You can also upload the model to the Konfuzio app or an on-prem setup.

```python tags=["skip-execution", "nbval-skip"]
from konfuzio_sdk.api import upload_ai_model

upload_ai_model(model_path=model_path, ai_type='extraction', category_id=category.id_)
```

### Extraction AI Evaluation

This section assumes you have already trained an Extraction AI model and have the pickle file available. If you have not done so, please first complete [this](/sdk/tutorials/rf-extraction-ai/) tutorial.

In this example we will see how we can evaluate a trained `RFExtractionAI` model. The model in the example is trained to extract data from payslip Documents. 

Start by loading the model:

```python tags=["skip-execution", "nbval-skip"]
from konfuzio_sdk.trainer.information_extraction import RFExtractionAI

pipeline = RFExtractionAI.load_model(MODEL_PATH)
```

Run the evaluation of the Extraction AI and check the metrics:
```python tags=["skip-execution", "nbval-skip"]
evaluation = pipeline.evaluate_full()
print(f"Full evaluation F1 score: {evaluation.f1()}")
print(f"Full evaluation recall: {evaluation.recall()}")
print(f"Full evaluation precision: {evaluation.precision()}")
```

You can also get the evaluation of the Tokenizer alone:
```python tags=["skip-execution", "nbval-skip"]
evaluation = pipeline.evaluate_tokenizer()
print(f"Tokenizer evaluation F1 score: {evaluation.tokenizer_f1()}")
```

It is also possible to get the evaluation of the Label classifier (given perfect tokenization).
```python tags=["skip-execution", "nbval-skip"]
evaluation = pipeline.evaluate_clf()
print(f"Label classifier evaluation F1 score: {evaluation.clf_f1()}")
```

Lastly, you can get the evaluation of the LabelSet (given perfect Label classification).
```python tags=["skip-execution", "nbval-skip"]
evaluation = pipeline.evaluate_clf()
print(f"Label Set evaluation F1 score: {evaluation.f1()}")
```

### Conclusion
This tutorial provided a comprehensive guide to building and deploying custom models for data extraction using Konfuzio. We covered various topics, including training a custom Extraction AI, evaluating model performance, and creating a practical example for extracting dates from Documents.

By following the steps outlined in this tutorial, you should now have the knowledge and tools to develop your own custom Extraction AIs tailored to your specific use cases. Additionally, we explored how to save and upload models to the Konfuzio Server for deployment in a real-world setting.

With this newfound understanding, you can continue to explore and enhance your skills in information extraction, enabling you to extract valuable insights from unstructured text data efficiently and effectively. 


### What's next?
- <a href="/sdk/tutorials/upload-your-ai">Learn how to upload your custom extraction model</a>
- <a href="/sdk/tutorials/async_upload_with_callback">Pull Documents Uploaded Asynchronously with a Webhook</a>