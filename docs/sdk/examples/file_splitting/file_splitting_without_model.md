## Split multi-file Document into Separate files without training a model

Let's see how to use the `konfuzio_sdk` to automatically split documents consisting of 
several files. We will be using a pre-built class SplittingAI. The class implements a context-aware rule-based logic 
that requires no training.

Note: the approach only works within a single Category.

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.file_splitting import SplittingAI

# initialize a Project and fetch a test Document of your choice
project = Project(id_=YOUR_PROJECT_ID)
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)

# initialize a class
splitting_ai = SplittingAI(project_id=project.id_, category_id=test_document.category.id_)

# gather Spans unique for the first Pages of the Documents in a chosen Category 
unique_spans = splitting_ai.train()

# propose a list of sub-Documents. If a test Document consists of a single file, return will consist of a list with the 
# same Document.
split_documents = splitting_ai.propose_split_documents(test_document, unique_spans)
```