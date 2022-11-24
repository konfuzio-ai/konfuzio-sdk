## Split Multi-File Documents into Separate Files

Let's see how to use the `konfuzio_sdk` to create a pipeline for automatically splitting documents consisting of 
several files.

```python
import pickle

from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.file_splitting import FileSplittingModel, SplittingAI

# initialize your Project and the FileSplittingModel

project = Project(id_=YOUR_PROJECT_ID)
file_splitting_model = FileSplittingModel(project_id=project.id_)
file_splitting_model.train_data = project.documents
file_splitting_model.test_data = project.test_documents

# preprocess train and test data

train_img_data, train_txt_data, test_img_data, test_txt_data, train_labels, test_labels, txt_input_shape = file_splitting.FileSplittingModel._prepare_visual_textual_data(
    train_documents,
    test_documents,
    bert_model,
    bert_tokenizer)

# initialize a fusion model

model = FileSplittingModel.init_model(txt_input_shape)

# train a model on the data from the project and save the result

model.fit([train_img_data, train_txt_data], train_labels, epochs=10, verbose=1)
pickler = open(project.model_folder + '/fusion.pickle', "wb")
pickle.dump(model, pickler)
pickler.close()

# evaluate the model

loss, acc = model.evaluate([test_img_data, test_txt_data], test_labels, verbose=0)
print('Accuracy: {}'.format(acc * 100))
precision, recall, f1 = FileSplittingModel.calculate_metrics(model,
                                                             test_img_data,
                                                             test_txt_data,
                                                             test_labels)
print('\n Precision: {} \n Recall: {} \n F1-score: {}'.format(precision, recall, f1))

# initialize SplittingAI with the model files saved during previous step
splitting_ai = SplittingAI(model_path=project.model_folder + '/fusion.pickle')

# run a prediction on a single document – returns a list of resulting subdocuments; 
# if no several documents are detected, returns a list with the original document

prediction = splitting_ai.propose_split_documents(test_documents[0])
```