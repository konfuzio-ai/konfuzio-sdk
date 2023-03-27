"""Test a code example for the File Splitting section of the documentation."""
import logging

from typing import List

from konfuzio_sdk.data import Page, Category, Project
from konfuzio_sdk.trainer.file_splitting import AbstractFileSplittingModel
from konfuzio_sdk.trainer.file_splitting import SplittingAI
from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel
from konfuzio_sdk.trainer.information_extraction import load_model
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer

from variables import YOUR_PROJECT_ID

logger = logging.getLogger(__name__)
YOUR_DOCUMENT_ID = 44865
[List, Page, Category, AbstractFileSplittingModel]  # for referencing in the imports and passing the linting

# initialize a Project and fetch a test Document of your choice

project = Project(id_=YOUR_PROJECT_ID)
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)

# initialize a Context Aware File Splitting Model and fit it

file_splitting_model = ContextAwareFileSplittingModel(categories=project.categories, tokenizer=ConnectedTextTokenizer())
# to run a Multimodal File Splitting Model instead, replace the line above with the following lines. note that training
# a Multimodal File Splitting Model can take longer that Context Aware File Splitting Model.
#
# from konfuzio_sdk.trainer.file_splitting import MultimodalFileSplittingModel
# file_splitting_model = MultimodalFileSplittingModel(categories=project.categories)

# for an example run, you can take only a slice of training documents to make fitting faster
file_splitting_model.documents = file_splitting_model.documents[:10]

file_splitting_model.fit(allow_empty_categories=True)

# save the model
save_path = file_splitting_model.save(include_konfuzio=True)

# run the prediction
for page in test_document.pages():
    pred = file_splitting_model.predict(page)
    if pred.is_first_page:
        print('Page {} is predicted as the first.'.format(page.number))
    else:
        print('Page {} is predicted as the non-first.'.format(page.number))

# usage with the Splitting AI – you can load a pre-saved model or pass an initialized instance as the input
# in this example, we load a previously saved one
model = load_model(save_path)

# initialize the Splitting AI
splitting_ai = SplittingAI(model)

# Splitting AI is a more high-level interface to Context Aware File Splitting Model and any other models that can be
# developed for File Splitting purposes. It takes a Document as an input, rather than individual Pages, because it
# utilizes page-level prediction of possible split points and returns Document or Documents with changes depending on
# the prediction mode.

# Splitting AI can be run in two modes: returning a list of Sub-Documents as the result of the input Document
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