# What is the Konfuzio SDK?

## Overview

The Open Source Konfuzio Software Development Kit (Konfuzio SDK) provides a Python API to build custom document processes. For a quick introduction to the SDK, check out the [Get Started](get_started.html) section. Review the release notes and the source code on [GitHub](https://github.com/konfuzio-ai/konfuzio-sdk/releases).

| Section                                                           | Description                                                                                                 |
|-------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| [Get Started](get_started.html)                                   | Learn more about the Konfuzio SDK and how it works.                                                         |
| [Tutorials](tutorials.html)                                       | Learn how to build your first document extraction pipeline, speed up your annotation process and many more. |
| [Explanations](explanations.html)                                 | Here are links to teaching material about the Konfuzio SDK.                                                 |
| [API Reference](sourcecode.html)                                  | Get to know all major Data Layer concepts of the Konfuzio SDK.                                              |
| [Contribution Guide](contribution.html)                           | Learn how to contribute, run the tests locally, and submit a Pull Request.                                  |
| [Changelog](https://github.com/konfuzio-ai/konfuzio-sdk/releases) | Review the release notes and the source code of the Konfuzio SDK.                                           |

## Customizing document processes with the Konfuzio SDK

For documentation about how to train and evaluate document understanding AIs, as well as extract new documents using 
the Konfuzio Server web interface, please see our [Konfuzio Guide](https://help.konfuzio.com/tutorials/quickstart/index.html).

If you need to **add custom functionality** to the document processes of the Konfuzio Server, the Konfuzio SDK 
is the tool for you. You can customize pipelines for automatic document Categorization, File Splitting, and Extraction.

.. note::
  Customizing document AI pipelines with the Konfuzio SDK requires a self-hosted installation of the Konfuzio Server.

### Three steps to customize document AI pipelines

1. The Konfuzio SDK defines abstract Python classes and interfaces for [Categorization](sourcecode.html#categorization-ai), [File Splitting](sourcecode.html#file-splitting-ai), and [Extraction](sourcecode.html#extraction-ai) AI pipelines. 
By implementing the abstract methods in a custom subclass, custom behaviours can be defined for each AI pipeline. Make
sure to familiarize yourself with the [Data Layer concepts](explanations.html#data-layer-concepts) so that you can 
manipulate data effectively within your custom pipelines.

2. All AIs inherit from [BaseModel](sourcecode.html#base-model), which provides `BaseModel.save` to generate a pickle file, 
to be directly uploaded to the Konfuzio Server (see [Upload Extraction or Category AI to target instance](https://help.konfuzio.com/tutorials/migrate-trained-ai-to-an-new-project-to-annotate-documents-faster/index.html#upload-extraction-or-category-ai-to-target-instance)). 

4. Finally, activating the uploaded AI on the web interface will enable the custom pipeline on your self-hosted installation.

### Customize Extraction AI

Any Custom [Extraction AI](sourcecode.html#extraction-ai) (derived from the Konfuzio `Trainer` class) should implement 
the following interface:
```python
from konfuzio_sdk.trainer.information_extraction import Trainer
from konfuzio_sdk.data import Document
from typing import List

class CustomExtractionAI(Trainer):

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

For a more in depth tutorial about the usage of Extraction AIs in the SDK see 
[Train a Konfuzio SDK Model to Extract Information From Payslip Documents](examples/examples.html#train-a-konfuzio-sdk-model-to-extract-information-from-payslip-documents).

### Customize Categorization AI

Any custom [Categorization AI](sourcecode.html#categorization-ai) (derived from the Konfuzio `AbstractCategorizationModel` class)  
should implement the following interface:
```python
from konfuzio_sdk.trainer.document_categorization import AbstractCategorizationModel
from konfuzio_sdk.data import Document
from typing import List

class CustomCategorizationAI(AbstractCategorizationModel):

    def __init__(self, *args, **kwargs):
        # initialize key variables required by the custom AI

    def fit(self):
        # Define architecture and training that the model undergoes, i.e. a NN architecture or a custom hardcoded logic
        # This method is allowed to be implemented as a no-op if you provide the trained model in other ways
    
    def _categorize_page(self, page: Page) -> Page:
        # define how the model assigns a Category to a Page
        # **NB:** The result of extraction must be the input Page with added Categorization attribute `Page.category`
```

Example usage of your Custom Categorization AI:
```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.information_extraction import load_model

# Initialize Project and provide the AI training and test data
project = Project(id_=YOUR_PROJECT_ID)  # see https://dev.konfuzio.com/sdk/get_started.html#example-usage

categorization_pipeline = CustomCategorizationAI(*args, **kwargs)
categorization_pipeline.categories = project.categories
categorization_pipeline.documents = [category.documents for category in categorization_pipeline.categories]
categorization_pipeline.test_documents = [category.test_documents() for category in categorization_pipeline.categories]

# Calculate features and train the AI
categorization_pipeline.fit()

# Evaluate the AI
data_quality = categorization_pipeline.evaluate(use_training_docs=True)
ai_quality = categorization_pipeline.evaluate(use_training_docs=False)

# Categorize a Document
document = self.project.get_document_by_id(YOUR_DOCUMENT_ID)
categorization_result = categorization_model.categorize(document=document)
for page in categorization_result.pages():
    print(f"Found category {page.category} for {page}")
print(f"Found category {categorization_result.category} for {result_doc}")

# Save and load a pickle file for the model
pickle_model_path = categorization_pipeline.save(output_dir=project.model_folder, include_konfuzio=True)
categorization_pipeline_loaded = load_model(pickle_model_path)
```

For a more in depth tutorial about the usage of Categorization AIs in the SDK see 
[Tutorials - Document Categorization](tutorials.html#document-categorization).

### Customize File Splitting AI

Any Custom [File Splitting AI](sourcecode.html#file-splitting-ai) (derived from the Konfuzio `AbstractFileSplittingModel` class) 
should implement the following interface:
```python
from konfuzio_sdk.trainer.file_spltting import AbstractFileSplittingModel
from konfuzio_sdk.data import Document, Page
from typing import List

class CustomFileSplittingModel(AbstractFileSplittingModel):

    def __init__(self, *args, **kwargs):
        # initialize key variables required by the custom AI

    def fit(self):
        # Define architecture and training that the model undergoes, i.e. a NN architecture or a custom hardcoded logic
        # This method is allowed to be implemented as a no-op if you provide the trained model in other ways

    def predict(self, page: Page) -> Page:
        # Define how the model determines a split point for a Page
        # **NB:** The classification needs to be ran on the Page level, not the Document level – the result of 
        # classification is reflected in `is_first_page` attribute value, which is unique to the Page class and is not 
        # present in Document class. Pages with `is_first_page = True` become splitting points, thus, each new 
        # sub-Document has a Page predicted as first as its starting point.
```

Example usage of your Custom File Splitting AI:
```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.file_splitting import SplittingAI
from konfuzio_sdk.trainer.information_extraction import load_model

# Initialize Project and provide the AI training and test data
project = Project(id_=YOUR_PROJECT_ID)  # see https://dev.konfuzio.com/sdk/get_started.html#example-usage

file_splitting_model = CustomFileSplittingModel(categories=project.categories, *args, *kwargs)
file_splitting_model.documents = [category.documents for category in categorization_pipeline.categories]
file_splitting_model.test_documents = [category.test_documents() for category in categorization_pipeline.categories]

# Calculate features and train the AI
file_splitting_model.fit()

# Initialize the Splitting pipeline by providing the trained model
file_splitting_pipeline = SplittingAI(file_splitting_model)

# Evaluate the AI
data_quality = file_splitting_pipeline.evaluate(use_training_docs=True)
ai_quality = file_splitting_pipeline.evaluate(use_training_docs=False)

# Suggest Page splits for a Document
document = project.get_document_by_id(YOUR_DOCUMENT_ID)
splitting_results = file_splitting_pipeline.propose_split_documents(document, return_pages=True)
for page in splitting_results.pages():
    if page.is_first_page:
        print(f'{page} is a suggested split point for {document}')

# Save and load a pickle file for the model
pickle_model_path = file_splitting_model.save()
file_splitting_model_loaded = load_model(pickle_model_path)
file_splitting_pipeline_loaded = SplittingAI(file_splitting_model_loaded)
```

For a more in depth tutorial about the usage of File Splitting AIs in the SDK see 
[Splitting for multi-Document files: Step-by-step guide](tutorials.html#splitting-for-multi-document-files-step-by-step-guide).

### Other use cases around documents

The following is a simple example of something you can build using the Konfuzio SDK.

#### Build your own PDF Form Generator

A simple implementation can look like the following.

##### 1. Upload a PDF Form and create empty annotations where you want to fill in data

.. image:: home/pdf_form_gen_example.png
   :scale: 40%

##### 2. Use the SDK to fill in values in the PDF

Example implementation:
```python
import io

from PyPDF2 import PdfFileWriter, PdfFileReader
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from konfuzio_sdk.data import Project, Document, Page, Annotation


def render(document_id, project_id, values):
    my_project = Project(id_=project_id, update=True)
    document: Document = my_project.get_document_by_id(document_id)

    # read your existing PDF
    existing_pdf = PdfFileReader(open(document.get_file(), "rb"))
    output = PdfFileWriter()

    for page_index, page in enumerate(document.pages()):
        packet = io.BytesIO()
        my_canvas = canvas.Canvas(packet)
        my_canvas.setPageSize((page.width, page.height))
        for annotation: Annotation in document.annotations():
            if annotation.selection_bbox.get('page_index') == page_index:
                text_value = values.get(annotation.label.name, '')
                textobject = my_canvas.beginText()
                # Set text location (x, y)
                textobject.setTextOrigin(annotation.x0, annotation.y0)
                # Change text color
                textobject.setFillColor(colors.black)
                # Write text
                textobject.textLine(text=text_value)
                # Write text to the canvas
                my_canvas.drawText(textobject)
        my_canvas.save()

        # move to the beginning of the StringIO buffer
        packet.seek(0)

        # create a new PDF with Reportlab
        new_pdf = PdfFileReader(packet)

        page = existing_pdf.getPage(page_index)
        page.mergePage(new_pdf.getPage(0))
        output.addPage(page)

    return output


if __name__ == '__main__':
    values = {'Name': 'Valerie', 'Kurs': 'Pilates', 'Startdatum': '15.01.2022', 'Enddatum': '30.03.2022'}
    # To find your Project ID and Document ID see https://dev.konfuzio.com/sdk/get_started.html#example-usage
    project_id = YOUR_PROJECT_ID
    document_id = YOUR_DOCUMENT_ID

    output = render(document_id=document_id, project_id=project_id, values=values)
    # finally, write "output" to a real file
    outputStream = open(f"{document_id}.pdf", "wb")
    output.write(outputStream)
    outputStream.close()
```

The resulting filled in PDF form will look like this:

.. image:: home/pdf_form_gen_example_compiled.png
   :scale: 40%
