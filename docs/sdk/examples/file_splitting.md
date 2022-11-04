## Split Multi-File Documents into Separate Files

Let's see how to use the `konfuzio_sdk` to create a pipeline for automatically splitting documents consisting of 
several files.

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer import file_splitting

# train a file-splitting model on Documents from your Project 

project = Project(id_=YOUR_PROJECT_ID)
train_documents = project.documents
test_documents = project.test_documents

# initialize LegalBERT

bert_model, bert_tokenizer = file_splitting.FileSplittingModel.init_bert()

# preprocess train and test data

train_img_data, train_txt_data, test_img_data, test_txt_data, train_labels, test_labels, txt_input_shape = file_splitting.FileSplittingModel.prepare_visual_textual_data(
                                                                                                                                                                        train_documents,
                                                                                                                                                                        test_documents,
                                                                                                                                                                        bert_model,
                                                                                                                                                                        bert_tokenizer)

# initialize a fusion model

model = file_splitting.FileSplittingModel.init_model(txt_input_shape)

# train a model on the data from the project and save the result

model.fit([train_img_data, train_txt_data], train_labels, epochs=10, verbose=1)
model.save(project.model_folder + '/fusion.h5')

# evaluate the model

loss, acc = model.evaluate([test_img_data, test_txt_data], test_labels, verbose=0)
print('Accuracy: {}'.format(acc * 100))
precision, recall, f1 = file_splitting.FileSplittingModel.calculate_metrics(model, 
                                                                            test_img_data, 
                                                                            test_txt_data,
                                                                            test_labels)
print('\n Precision: {} \n Recall: {} \n F1-score: {}'.format(precision, recall, f1))

# initialize SplittingAI with the model files saved during previous step

splitting_ai = file_splitting.SplittingAI(model_path=project.model_folder + '/fusion.h5')

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