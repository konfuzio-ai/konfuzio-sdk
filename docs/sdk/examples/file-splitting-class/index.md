## Splitting for multi-file Documents: Step-by-step guide

### Intro

Not all multipage files that we process are always neatly scanned and organized. Sometimes we can have more than one actual 
Document in a stream of pages; such files need to be properly processed and split into several independent Documents.

.. image:: /_static/img/multi_file_document_example.png

_Multi-file Document Example_

In this post, we will look at a simple way to implement an algorithm for searching common features between 
Documents/Pages which can be used for splitting a multi-file Document into the sub-Documents. Our approach is based on 
an assumption that we can go through all Pages of the Document and define splitting points. By splitting points, we mean 
Pages that are similar in contents to the first Pages in the Documents. 
Note: this approach only works with the Documents in the same language and gathering happens in each Category 
independently. 

If you are unfamiliar with the SDK's main concepts (like Page or Span), you can get to know them on the Quickstart page.


### Quick explanation

First step for implementation is "training": we tokenize the Document (meaning that we split its text into parts, more 
specifically to this tutorial – into strings without line breaks) and gather exclusive strings from Spans – the parts of 
the text in the Page – each of which is present on every first Page of each Documents in the training data. 

Then, for every input Document's Page, we determine if it's first or not by going through its strings and comparing them 
to the set of strings collected in the first step. If we have at least one string of intersection between the current Page 
and the strings from the first step, we believe it is the first Page.

Note that the more Documents we use in the "training" stage, the less intersecting strings we are likely to find, so if at
the end of the tutorial you find that your first-Page strings set is empty, try using a slice of the dataset instead of 
the whole like it is done in the example below. However, usually when ran on Documents within same Category, this 
algorithm should not return an empty set; if that is the case, you might want to check if your data is consistent (i.e. 
not in different languages, no occurrences of other Categories).

### Step-by-step explanation

In this section, we'll go through steps imitating initialization of `ContextAwareFileSplittingModel` class which you can 
find in the full code block in the lower part of this page. A class itself is already implemented and can be imported via
`from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel`.

Let's start with making all the necessary imports and initializing the class of `ContextAwareFileSplittingModel`:

```python
import logging

from typing import List

from konfuzio_sdk.data import Page, Category
from konfuzio_sdk.trainer.file_splitting import AbstractFileSplittingModel
from konfuzio_sdk.trainer.information_extraction import load_model
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer

class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    def __init__(self, categories: List[Category], tokenizer, *args, **kwargs):
        super().__init__()
        self.name = self.__class__.__name__
        if not len(categories):
            raise ValueError("Cannot initialize ContextAwareFileSplittingModel on an empty list.")
        for category in categories:
            if not type(category) == Category:
                raise ValueError("All elements of the list have to be Categories.")
            if not len(category.documents()):
                raise ValueError(f'{category} does not have Documents and cannot be used for training.')
            if not len(category.test_documents()):
                raise ValueError(f'{category} does not have test Documents.')
        projects = set([category.project for category in categories])
        if len(projects) > 1:
            raise ValueError("All Categories have to belong to the same Project.")
        self.categories = categories
        self.project = self.categories[0].project  # we ensured that at least one Category is present
        self.output_dir = self.project.model_folder
        self.documents = [document for category in self.categories for document in category.documents()]
        self.test_documents = [document for category in self.categories for document in category.test_documents()]
        self.tokenizer = tokenizer
        self.path = None
        self._used_tokenizer = None
```
The class inherits from `AbstractFileSplittingModel`, so we run `super().__init__()` for proper inheritance of the 
attributes. Then we conduct checks to ensure that a list of Categories passed into `categories` is fitting all the 
criteria. Project, output directory, Documents and test Documents are taken from Categories passed earlier. `tokenizer` 
will be used for going through the Document's text and separating it into Spans. We will use it to process training and 
testing Documents, as well as any Document that will undergo splitting. This is done to ensure that texts in all the 
Documents are split using the same logic (particularly tokenization by separating on `\n` whitespaces by 
ConnectedTextTokenizer, which is used in the example in the end of the page) and it will be possible to find common 
Spans. `path` is a full path of the model-to-be-saved. Note: if you run fitting with one tokenizer and then reassign it
within the same instance of a model, all previously gathered strings will be deleted and replaced by new ones. 

An example of how ConnectedTextTokenizer works:
```python
# before tokenization
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)
test_document.text

# output: "This is an example text. \n It has several lines. \n Here it finishes."

test_document.spans()

# output: []

test_document = tokenizer.tokenize(test_document)

# after tokenization
test_document.spans()

# output: [Span (0, 24), Span(25, 47), Span(48, 65)]

test_document.spans[0].offset_string

# output: "This is an example text. "
```

A first method to define will be `fit()`. In the beginning, we check whether the current tokenizer is the same as the 
previously used one within the instance – this is needed because if strings gathered within differently tokenized 
Documents are mixed, it is unlikely that the prediction of the model runs successfully. `allow_empty_categories` allows
returning empty lists for Categories that haven't had any exclusive first-page strings found across their Documents
(meaning it would not be used in prediction).
```python
    def fit(self, allow_empty_categories: bool = False, *args, **kwargs):
        if not self._used_tokenizer:
            self._used_tokenizer = self.tokenizer
        else:
            if self.tokenizer != self._used_tokenizer:
                logger.warning(
                    "Assigned tokenizer does not correspond to the one previously used within this instance."
                    "All previously found exclusive first-page strings within each Category will be removed "
                    "and replaced with the newly generated ones."
                )
                for category in self.categories:
                    category._exclusive_first_page_strings = None
        for category in self.categories:
            if not category.exclusive_first_page_strings:
                category.collect_exclusive_first_page_strings(self.tokenizer)
                if allow_empty_categories:
                    logger.warning(
                        f'No exclusive first-page strings were found for {category}, so it will not be used '
                        f'at prediction.'
                    )
                else:
                    raise ValueError(f'No exclusive first-page strings were found for {category}.')
```

Lastly, we define `predict()` method. A Page is accepted as an input and its Span set is checked for containing 
first-page strings for each of the Categories. If there has been at least one intersection, a Page is predicted to be 
first; if there's no such intersections, it's predicted non-first. 

```python
    def predict(self, page: Page) -> Page:
        for category in self.categories:
            if not category.exclusive_first_page_strings:
                raise ValueError(f"Cannot run prediction as {category} does not have exclusive_first_page_strings.")
        page.is_first_page = False
        for category in self.categories:
            intersection = {span.offset_string for span in page.spans()}.intersection(
                category.exclusive_first_page_strings
            )
            if len(intersection) > 0:
                page.is_first_page = True
                break
        return page
```

A quick example of the class's usage:

```python
# initialize a Project and fetch a test Document of your choice
project = Project(id_=YOUR_PROJECT_ID)
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)

# initialize a ContextAwareFileSplittingModel and fit ut

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

# SplittingAI can be ran in two modes: returning a list of sub-Documents as the result of the input Document
# splitting or returning a copy of the input Document with Pages predicted as first having an attribute
# "is_first_page". The flag "return_pages" has to be True for the latter; let's use it
new_document = splitting_ai.propose_split_documents(test_document, return_pages=True)
```
Note that any custom FileSplittingAI (derived from `AbstractFileSplittingModel` class) requires having the following 
methods implemented:
- `__init__` to initialize key variables required by the custom AI;
- `fit` to define architecture and training that the model undergoes, i.e. a certain NN architecture or a custom 
- hardcoded logic
- `predict` to define how the model classifies Pages as first or non-first. **NB:** the classification needs to be ran on 
the Page level, not the Document level – the result of classification is reflected in `is_first_page` attribute value, which
is unique to the Page class and is not present in Document class. Pages with `is_first_page = True` become splitting 
points, thus, each new sub-Document has a Page predicted as first as its starting point.
- `temp_pkl_file_path` and `pkl_file_path` to define temporary and final path for saving a model with compression.
- `lose_weight` to remove all Documents before saving, used in `reduce_model_weight`.
- `reduce_model_weight` to make a model smaller during saving.
- `ensure_model_memory_usage_within_limit` to ensure that a model is not exceeding allowed max_ram.
- `restore_category_documents_for_eval` to restore Documents previously deleted at reducing weight step, in case 
evaluation is needed.

Full code:

```python
import logging

from typing import List

from konfuzio_sdk.data import Page, Category
from konfuzio_sdk.trainer.file_splitting import AbstractFileSplittingModel
from konfuzio_sdk.trainer.information_extraction import load_model
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer

logger = logging.getLogger(__name__)

class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    """Fallback definition of a File Splitting Model."""

    def __init__(self, categories: List[Category], tokenizer, *args, **kwargs):
        super().__init__()
        self.name = self.__class__.__name__
        if not len(categories):
            raise ValueError("Cannot initialize ContextAwareFileSplittingModel on an empty list.")
        for category in categories:
            if not type(category) == Category:
                raise ValueError("All elements of the list have to be Categories.")
            if not len(category.documents()):
                raise ValueError(f'{category} does not have Documents and cannot be used for training.')
            if not len(category.test_documents()):
                raise ValueError(f'{category} does not have test Documents.')
        projects = set([category.project for category in categories])
        if len(projects) > 1:
            raise ValueError("All Categories have to belong to the same Project.")
        self.categories = categories
        self.project = self.categories[0].project  # we ensured that at least one Category is present
        self.output_dir = self.project.model_folder
        self.documents = [document for category in self.categories for document in category.documents()]
        self.test_documents = [document for category in self.categories for document in category.test_documents()]
        self.tokenizer = tokenizer
        self.path = None
        self._used_tokenizer = None

    def fit(self, allow_empty_categories: bool = False, *args, **kwargs):
        if not self._used_tokenizer:
            self._used_tokenizer = self.tokenizer
        else:
            if self.tokenizer != self._used_tokenizer:
                logger.warning(
                    "Assigned tokenizer does not correspond to the one previously used within this instance."
                    "All previously found exclusive first-page strings within each Category will be removed "
                    "and replaced with the newly generated ones."
                )
                for category in self.categories:
                    category._exclusive_first_page_strings = None
        for category in self.categories:
            if not category.exclusive_first_page_strings:
                category.collect_exclusive_first_page_strings(self.tokenizer)
                if allow_empty_categories:
                    logger.warning(
                        f'No exclusive first-page strings were found for {category}, so it will not be used '
                        f'at prediction.'
                    )
                else:
                    raise ValueError(f'No exclusive first-page strings were found for {category}.')

    def predict(self, page: Page) -> Page:
        for category in self.categories:
            if not category.exclusive_first_page_strings:
                raise ValueError(f"Cannot run prediction as {category} does not have exclusive_first_page_strings.")
        page.is_first_page = False
        for category in self.categories:
            intersection = {span.offset_string for span in page.spans()}.intersection(
                category.exclusive_first_page_strings
            )
            if len(intersection) > 0:
                page.is_first_page = True
                break
        return page

# initialize a Project and fetch a test Document of your choice
project = Project(id_=YOUR_PROJECT_ID)
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)

# initialize a ContextAwareFileSplittingModel and fit ut

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

# SplittingAI can be ran in two modes: returning a list of sub-Documents as the result of the input Document
# splitting or returning a copy of the input Document with Pages predicted as first having an attribute
# "is_first_page". The flag "return_pages" has to be True for the latter; let's use it
new_document = splitting_ai.propose_split_documents(test_document, return_pages=True)
```