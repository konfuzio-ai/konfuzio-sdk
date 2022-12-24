## Split multi-file Document into Separate files without training a model

Let's see how to use the `konfuzio_sdk` to automatically split documents consisting of 
several files. We will be using a pre-built class SplittingAI. The class implements a context-aware rule-based logic 
that requires no training.

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel, SplittingAI
from konfuzio_sdk.trainer.information_extraction import load_model

# initialize a Project and fetch a test Document of your choice
project = Project(id_=YOUR_PROJECT_ID)
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)

# initialize a ContextAwareFileSplittingModel and define its attributes

file_splitting_model = ContextAwareFileSplittingModel()
file_splitting_model.categories = project.categories
file_splitting_model.documents = [document
                                  for category in file_splitting_model.categories
                                  for document in category.documents()
                                  ]
file_splitting_model.test_documents = [document
                                       for category in file_splitting_model.categories
                                       for document in category.test_documents()
                                       ]
file_splitting_model.tokenizer = ConnectedTextTokenizer()

# gather Spans unique for the first Pages of the Documents 
# the gathered Spans are saved to later be reused in the SplittingAI
file_splitting_model.fit()

# save the gathered Spans
file_splitting_model.save(project.model_folder)

# usage with the SplittingAI – you can load a pre-saved model or pass an initialized instance as the input
# in this example, we load a previously saved one
model = load_model(project.model_folder)

# initialize the SplittingAI
splitting_ai = SplittingAI(model)

# SplittingAI can be ran in two modes: returning a list of sub-Documents as the result of the input Document
# splitting or returning a copy of the input Document with Pages predicted as first having an attribute
# "is_first_page". The flag "return_pages" has to be True for the latter; let's use it.
new_document = splitting_ai.predict(test_document, return_pages=True)
```