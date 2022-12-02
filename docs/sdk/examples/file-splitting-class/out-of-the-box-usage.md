## Split multi-file Document into Separate files without training a model

Let's see how to use the `konfuzio_sdk` to automatically split documents consisting of 
several files. We will be using a pre-built class SplittingAI. The class implements a context-aware rule-based logic 
that requires no training.


```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel, SplittingAI

# initialize a Project and fetch a test Document of your choice
project = Project(id_=YOUR_PROJECT_ID)
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)

# initialize a ContextAwareFileSplittingModel and define its attributes

file_splitting_model = ContextAwareFileSplittingModel()
file_splitting_model.categories = project.categories
file_splitting_model.train_data = [document 
                                   for category in file_splitting_model.categories 
                                   for document in category.documents()
                                   ]
file_splitting_model.test_data = [document 
                                   for category in file_splitting_model.categories 
                                   for document in category.test_documents()
                                   ]
file_splitting_model.tokenizer = ConnectedTextTokenizer()

# gather Spans unique for the first Pages of the Documents 
# the gathered Spans are saved to later be reused in the SplittingAI
file_splitting_model.fit()

# initialize SplittingAI 
splitting_ai = SplittingAI(project_id=project.id_)

# process the test Document because it needs to be tokenized by the same Tokenizer as the Documents used by 
# ContextAwareFileSplittingModel for gathering Spans unique for the first Pages. deepcopy is used to remove all previous
# Annotations.
test_document = file_splitting_model.tokenizer.tokenize(test_document)

# propose a list of sub-Documents. If a test Document consists of a single file, return will consist of a list with the 
# same Document.
split_documents = splitting_ai.propose_split_documents(test_document)
```