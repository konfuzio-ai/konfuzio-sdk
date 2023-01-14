# What is the Konfuzio SDK?

## Overview

The Open Source Konfuzio Software Development Kit (Konfuzio SDK) provides a Python API to build custom document processes. For a quick introduction to the SDK, check out the [Quickstart](https://dev.konfuzio.com/sdk/home/index.html) section. Review the release notes and the source code on [GitHub](https://github.com/konfuzio-ai/konfuzio-sdk/releases).

| Section                                                           | Description                                                                                                 |
|-------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| [Get Started](get_started.html)                                   | Learn more about the Konfuzio SDK and how it works.                                                         |
| [Tutorials](examples/examples.html)                               | Learn how to build your first document extraction pipeline, speed up your annotation process and many more. |
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

### General customization process overview

The Konfuzio SDK defines abstract Python classes and interfaces for [Categorization](sourcecode.html#document-categorization), [File Splitting](sourcecode.html#file-splitting), and [Extraction](sourcecode.html#extraction-ai) AI pipelines. 
By implementing the abstract methods, custom behaviours can be defined for each AI pipeline.

All AIs inherit from [BaseModel](sourcecode.html#basemodel), which provides `BaseModel.save` to generate a pickle file, 
which can be directly uploaded to the Konfuzio Server.

See [Upload Extraction or Category AI to target instance](../web/on_premises.html#upload-extraction-or-category-ai-to-target-instance). 
Finally, activating the uploaded AI on the web interface will enable the custom pipeline on your self-hosted installation.

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers tags lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;url&quot;:&quot;https://raw.githubusercontent.com/konfuzio-ai/konfuzio-sdk/9869-dev-documentation/docs/sdk/home/SDK_Server_components.drawio&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/embed2.js?&fetch=https%3A%2F%2Fraw.githubusercontent.com%2Fkonfuzio-ai%2Fkonfuzio-sdk%2F9869-dev-documentation%2Fdocs%2Fsdk%2Fhome%2FSDK_Server_components.drawio"></script>

For more architectural details about how the Konfuzio Server and the Konfuzio SDK are integrated, see 
[Architecture SDK to Server](explanations.html#architecture-sdk-to-server).

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

    def feature_function(self, documents: List[Document]):
        # Create training and test data for the AI out of the Training and Test Documents in the provided Category
        # This method is allowed to be implemented as a no-op if you create the features in other ways

    def fit(self):
        # Define architecture and training that the model undergoes, i.e. a NN architecture or a custom hardcoded logic
        # This method is allowed to be implemented as a no-op if you provide the trained model in other ways

    def extract(self, document: Document) -> Document:
        # define how the model extracts information from Documents
        # **NB:** The result of extraction must be a copy of the input Document with added Annotations attribute `Document._annotations`
```

Example usage of your Custom Extraction AI:
```python
from konfuzio_sdk.data import Project

# Initialize Project and provide the AI training and test data
project = Project(id_=YOUR_PROJECT_ID)

extraction_pipeline = CustomExtractionAI(*args, **kwargs)
extraction_pipeline.category = project.get_category_by_id(id_=YOUR_CATEGORY_ID)
extraction_pipeline.documents = extraction_pipeline.category.documents()
extraction_pipeline.test_documents = extraction_pipeline.category.test_documents()

# Calculate features and train the AI
extraction_pipeline.feature_function(documents=extraction_pipeline.documents)
extraction_pipeline.fit()

# Evaluate the AI
data_quality = extraction_pipeline.evaluate_full(use_training_docs=True)
ai_quality = extraction_pipeline.evaluate_full(use_training_docs=False)

# Extract a Document
document = self.project.get_document_by_id(YOUR_DOCUMENT_ID)
extraction_result = extraction_pipeline.extract(document=document)

# Save and load a pickle file for the model
pickle_model_path = extraction_pipeline.save(output_dir=project.model_folder, include_konfuzio=True)
extraction_pipeline_loaded = load_model(pickle_model_path)
```

### Customize Categorization AI

Any custom [Categorization AI](sourcecode.html#document-categorization) (derived from the Konfuzio `FallbackCategorizationModel` class)  
should implement the following interface:
```python
from konfuzio_sdk.trainer.document_categorization import FallbackCategorizationModel
from konfuzio_sdk.data import Document
from typing import List

class CustomCategorizationAI(FallbackCategorizationModel):

    def __init__(self, *args, **kwargs):
        # initialize key variables required by the custom AI

    def feature_function(self, documents: List[Document]):
        # Create training and test data for the AI out of the Training and Test Documents in the provided Categories
        # This method is allowed to be implemented as a no-op if you create the features in other ways

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

# Initialize Project and provide the AI training and test data
project = Project(id_=YOUR_PROJECT_ID)

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

### Customize File Splitting AI

Note that any custom FileSplittingAI (derived from `AbstractFileSplittingModel` class) requires having the following 
methods implemented:
- `__init__` to initialize key variables required by the custom AI;
- `fit` to define architecture and training that the model undergoes, i.e. a certain NN architecture or a custom 
- hardcoded logic
- `predict` to define how the model classifies Pages as first or non-first. **NB:** the classification needs to be ran on 
the Page level, not the Document level – the result of classification is reflected in `is_first_page` attribute value, which
is unique to the Page class and is not present in Document class. Pages with `is_first_page = True` become splitting 
points, thus, each new sub-Document has a Page predicted as first as its starting point.

Any Custom [File Splitting AI](sourcecode.html#file-splitting) (derived from the Konfuzio `AbstractFileSplittingModel` class) 
should implement the following interface:
```python
```

Example usage of your Custom File Splitting AI:
```python
```

### PDF Form Generator

TODO
