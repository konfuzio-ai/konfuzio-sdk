## Develop a Custom AI

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
the interface described in the following example, which shows a simple example of how to manipulate documents' data. 
For more details on how to manipulate such data, always refer to the explanation at 
[Data Layer Concepts](explanations.html#data-layer-concepts).
```python
from copy import deepcopy
from typing import List

from konfuzio_sdk.trainer.information_extraction import Trainer
from konfuzio_sdk.data import Document, Annotation, Span
from konfuzio_sdk.tokenizer.regex import RegexTokenizer
from konfuzio_sdk.tokenizer.base import ListTokenizer


class CustomExtractionAI(Trainer):

    def __init__(self, *args, **kwargs):
        # Initialize key variables required by the custom AI.
        super.__init__(*args, **kwargs)

    def extract(self, document: Document) -> Document:
        # Define how the AI extracts information from Documents.

        # **NB:** The result of extraction must be a copy of the input Document.
        result_document = deepcopy(document)
    
        # You can access the text of the document with result_document.text
        # Example:
        # >>> result_document.text
        # "Name: John\nSurname: Doe\nA Software Company Ltd.\nOther text here"
        # Suppose we want to extract "John" and "Doe".
        # We can match that text using regular expressions:
        name_tokenizer = RegexTokenizer(regex=r"Name:[ ]([A-Z][a-z]+)")
        surname_tokenizer = RegexTokenizer(regex=r"Surname:[ ]([A-Z][a-z]+)")
        tokenizer = ListTokenizer(tokenizers=[name_tokenizer, surname_tokenizer])
        # The tokenizer will create Annotations objects within the document
        tokenizer.tokenize(result_document)
        
        # These Annotations will be the extraction results.
        # At the moment, these Annotations have no Label, which would exclude them from the extraction results.
        # We need to associate the proper Labels to each Annotation, assuming that these exist in our Project.
        name_label = self.project.get_label_by_name("Name")  # the self.project attribute is derived from Trainer
        surname_label = self.project.get_label_by_name("Surname")
        for annotation in result_document.annotations():
            for span in annotation.spans:
               # Each Annotation contains information about which tokenizer found it.
               # In this example, we associate the Label straighforwardly.
               # If your regex can produce false positives, you will want to apply some filtering logic here.
               if name_tokenizer in span.regex_matching:
                  annotation.label = name_label
                  break
               elif surname_tokenizer in span.regex_matching:
                  annotation.label = surname_label
                  break

        # Suppose we want to extract "A Software Company Ltd.", which does not have a clear regex pattern, but
        # we know it's always the third line in the Document. We can explicitly create an Annotation based on a 
        # substring of the Document's text.
        company_label = self.project.get_label_by_name("Company")
        company_substring = result_document.split('\n')[2]  # third line of the Document
        start_offset = result_document.find(company_substring)
        end_offset = start_offset + len(company_substring)
        _ = Annotation(document=result_document, label=company_label, spans=[Span(start_offset, end_offset)])

        # The resulting Document has 3 extractions. You can double check that they are there with:
        # >>> result_document.annotations(use_correct=False)
        # [
        #     Annotation Name (6, 10),
        #     Annotation Surname (20, 23),
        #     Annotation Company (24, 27)
        # ]
        return result_document
```

Example usage of the interface of your Custom Extraction AI:
```python
from konfuzio_sdk.data import Project, Document
from konfuzio_sdk.trainer.information_extraction import load_model
from konfuzio_sdk.evaluate import ExtractionEvaluation

# Initialize Project and provide the AI training and test data
project = Project(id_=YOUR_PROJECT_ID)  # also see https://dev.konfuzio.com/sdk/get_started.html#example-usage

extraction_pipeline = CustomExtractionAI(*args, **kwargs)
extraction_pipeline.category = project.get_category_by_id(id_=YOUR_CATEGORY_ID)

# Provide Training and Test Documents for evaluation
extraction_pipeline.documents = extraction_pipeline.category.documents()
extraction_pipeline.test_documents = extraction_pipeline.category.test_documents()

# Evaluate the AI
data_quality: ExtractionEvaluation = extraction_pipeline.evaluate_full(use_training_docs=True)
ai_quality: ExtractionEvaluation = extraction_pipeline.evaluate_full(use_training_docs=False)

# Extract a Document
document = self.project.get_document_by_id(YOUR_DOCUMENT_ID)
extraction_result: Document = extraction_pipeline.extract(document=document)
print(f"Extractions: {extraction_result.annotations(use_correct=False)}")

# Save and load a pickle file for the model
pickle_ai_path = extraction_pipeline.save(output_dir=project.model_folder, include_konfuzio=True)
extraction_pipeline_loaded = load_model(pickle_ai_path)
```
After saving the AI's pickle file, this can be uploaded to a Konfuzio Server instance to extract new Documents.

For an example notebook about the usage of Extraction AIs in the SDK see 
[Train a Konfuzio SDK Model to Extract Information From Payslip Documents](tutorials.html#train-a-konfuzio-sdk-model-to-extract-information-from-payslip-documents).

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
        super.__init__(*args, **kwargs)

    def _categorize_page(self, page: Page) -> Page:
        # Define how the AI assigns a Category to a Page.

        # The image of the Page is accessible as a Pillow Image object
        # Example:
        # >>> page.get_image()
        # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1764x3778 at 0x7F75D5396190>
        # The text of the Page is accessible as a string
        # Simplified example:
        # >>> page.text
        # "TAX INVOICE\n\nBilling Address: ADDRESS_HERE\n\nInvoice No: 02\nDate: 26.11.2019\nOTHER_INVOICE_INFORMATION"

        # Let's say you have a Project with invoices and delivery notes:
        # >>> self.categories
        # [Invoices (INVOICE_CATEGORY_ID), Delivery Notes (DELIVERY_NOTE_CATEGORY_ID)]
        # Your Custom AI predicts an Invoice (or Delivery Note) for this Page, according to some logic:
        predicted_category = self.categories[0]  # or self.categories[1]
        confidence = 1.0  # choose a confidence value between 0.0 and 1.0

        # **NB:** The result of categorization must be the input Page with added CategoryAnnotation object
        _ = CategoryAnnotation(category=predicted_category, confidence=confidence, page=page)
        return page
```

Example usage of the interface of your Custom Categorization AI:
```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.information_extraction import load_model
from konfuzio_sdk.evaluate import CategorizationEvaluation

# Initialize Project and provide the AI training and test data
project = Project(id_=YOUR_PROJECT_ID)  # see https://dev.konfuzio.com/sdk/get_started.html#example-usage

categorization_pipeline = CustomCategorizationAI(*args, **kwargs)
categorization_pipeline.categories = project.categories

# Provide Training and Test Documents for evaluation
categorization_pipeline.documents = [category.documents for category in categorization_pipeline.categories]
categorization_pipeline.test_documents = [category.test_documents() for category in categorization_pipeline.categories]

# Evaluate the AI
data_quality: CategorizationEvaluation = categorization_pipeline.evaluate(use_training_docs=True)
ai_quality: CategorizationEvaluation = categorization_pipeline.evaluate(use_training_docs=False)

# Categorize a Document
document = self.project.get_document_by_id(YOUR_DOCUMENT_ID)
categorization_result = categorization_model.categorize(document=document)
for page in categorization_result.pages():
    print(f"Found category {page.category} for {page}")
print(f"Found category {categorization_result.category} for {result_doc}")

# Save and load a pickle file for the model
pickle_ai_path = categorization_pipeline.save(output_dir=project.model_folder, include_konfuzio=True)
categorization_pipeline_loaded = load_model(pickle_ai_path)
```

To see more code examples about the usage of Categorization AIs in the SDK see 
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
        super.__init__(*args, **kwargs)

    def predict(self, page: Page) -> Page:
        # Define how the AI determines a split point for a Page.

        # The image of the Page is accessible as a Pillow Image object
        # Example:
        # >>> page.get_image()
        # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1764x3778 at 0x7F75D5396190>
        # The text of the Page is accessible as a string
        # Simplified example:
        # >>> page.text
        # "TAX INVOICE\n\nBilling Address: ADDRESS_HERE\n\nInvoice No: 02\nDate: 26.11.2019\nOTHER_INVOICE_INFORMATION"

        # **NB:** The split points need to be specified on the Page level by using a boolean flag to 
        # mark which Pages are the at the beginning of a new Document.
        # Thus, if your Custom AI predicts that this page is (not) a split point, according to some logic, then:
        page.is_first_page = True  # or False
        page.is_first_page_confidence = 1.0  # choose a confidence value between 0.0 and 1.0
        return page
```

Example usage of the interface of your Custom File Splitting AI:
```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.file_splitting import SplittingAI
from konfuzio_sdk.trainer.information_extraction import load_model

# Initialize Project and provide the AI training and test data
project = Project(id_=YOUR_PROJECT_ID)  # see https://dev.konfuzio.com/sdk/get_started.html#example-usage

file_splitting_model = CustomFileSplittingModel(categories=project.categories, *args, *kwargs)

# Provide Training and Test Documents for evaluation
file_splitting_model.documents = [category.documents for category in categorization_pipeline.categories]
file_splitting_model.test_documents = [category.test_documents() for category in categorization_pipeline.categories]

# Initialize the Splitting pipeline by providing the custom model
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
