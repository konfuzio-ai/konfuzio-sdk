## Split Multi-File Documents into Separate Files

Let's see how to use the `konfuzio_sdk` to create a pipeline for automatically splitting documents consisting of 
several files.

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer import file_splitting

project = Project(id_=YOUR_PROJECT_ID)
train_documents = project.documents
test_documents = project.test_documents

# train a file-splitting model on Documents from your Project 

# train a model for visual inputs

vgg16 = file_splitting.FileSplittingModel.vgg16_preprocess_and_train(train_documents,
                                                                     test_documents,
                                                                     0.5)
# initialize LegalBERT

bert_model, bert_tokenizer = file_splitting.FileSplittingModel.init_bert()

# prepare data for inputs of a file-splitting model 

Xtrain, Xtest, ytrain, ytest, input_shape = file_splitting.FileSplittingModel.prepare_mlp_inputs(train_documents, 
                                                                                                test_documents, 
                                                                                                0.5,
                                                                                                vgg16,
                                                                                                bert_tokenizer,
                                                                                                bert_model)
# run training, evaluation, and saving of the file-splitting model

model = file_splitting.FileSplittingModel.run_mlp(Xtrain, Xtest, ytrain, ytest, input_shape)

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