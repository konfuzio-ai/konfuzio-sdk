## Split Multi-File Documents into Separate Files

Let's see how to use the `konfuzio_sdk` to create a pipeline for automatically splitting documents consisting of 
several files. 

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer import file_splitting

project = Project(id_=YOUR_PROJECT_ID)
test_documents = project.test_documents

# train a file-splitting model on Documents from your Project 

fusion_model = file_splitting.FusionModel(project_id=project.id_, split_point=0.5)

# initialize SplittingAI with the model files saved during previous step

splitting_ai = file_splitting.SplittingAI(model_path=project.model_folder + '/splitting_ai_models.tar.gz')

# run a prediction on a single document – returns a list of resulting subdocuments; 
# if no several documents are detected, returns a list with the original document

prediction = splitting_ai.propose_split_documents(test_documents[0])

# iterate over the test documents to predict if any of them need to be split

split_documents = {}

for doc in test_documents:
    prediction = splitting_ai.propose_split_documents(doc)
    if len(prediction) > 1:
        split_documents[doc.id_] = prediction
```