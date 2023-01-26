## Split multi-file Document into Separate files without training a model

Let's see how to use the `konfuzio_sdk` to automatically split documents consisting of 
several files. We will be using a pre-built class SplittingAI. The class implements a context-aware rule-based logic 
that requires no training.

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel, SplittingAI
from konfuzio_sdk.trainer.information_extraction import load_model

project = Project(id_=YOUR_PROJECT_ID)
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)

# initialize a ContextAwareFileSplittingModel and fit it

file_splitting_model = ContextAwareFileSplittingModel(categories=project.categories, tokenizer=ConnectedTextTokenizer())
file_splitting_model.fit()

# save the model
file_splitting_model.output_dir = project.model_folder
file_splitting_model.save()

# run the prediction
for page in test_document.pages():
    pred = file_splitting_model.predict(page)
    if pred.is_first_page:
        print('Page {} is predicted as the first.'.format(page.number))
    else:
        print('Page {} is predicted as the non-first.'.format(page.number))

# usage with the SplittingAI – you can load a pre-saved model or pass an initialized instance as the input
# in this example, we load a previously saved one
model = load_model(project.model_folder)

# initialize the SplittingAI
splitting_ai = SplittingAI(model)

# SplittingAI is a more high-level interface to ContextAwareFileSplittingModel and any other models that can be 
# developed for file-splitting purposes. It takes a Document as an input, rather than individual Pages, because it 
# utilizes page-level prediction of possible split points and returns Document or Documents with changes depending on 
# the prediction mode.

# SplittingAI can be ran in two modes: returning a list of sub-Documents as the result of the input Document
# splitting or returning a copy of the input Document with Pages predicted as first having an attribute
# "is_first_page". The flag "return_pages" has to be True for the latter; let's use it
new_document = splitting_ai.propose_split_documents(test_document, return_pages=True)
print(new_document)
# output: [predicted_document]

for page in new_document[0].pages():
    if page.is_first_page:
        print('Page {} is predicted as the first.'.format(page.number))
    else:
        print('Page {} is predicted as the non-first.'.format(page.number))
```