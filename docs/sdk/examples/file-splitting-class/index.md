## Splitting for multi-file Documents: Step-by-step guide

### Intro

Not all multipage files that we process are always neatly scanned and organized. Sometimes we can have more than one actual 
Document in a stream of pages; such files need to be properly processed and split into several independent Documents

.. image:: /_static/img/multi_file_document_example.png

_Multi-file Document Example_

In this post, we will look at a simple way to implement an algorithm for searching common features between 
Documents/Pages which can be used for splitting a multi-file Document into the sub-Documents. Our approach is based on 
an assumption that we can go through all Pages of the Document and define splitting points. By splitting points, we mean 
Pages that are similar in contents to the first Pages in the Documents. 
Note: this approach only works with the Documents of the same Category and in the same language. 

If you are unfamiliar with the SDK's main concepts (like Page or Span), you can get to know them on the Quickstart page.


### Quick explanation

First step for implementation is "training": we tokenize the Document (meaning that we split its text into parts, more 
specifically to this tutorial – into strings without line breaks) and gather Spans – the parts of the text in the Page –
each of which is present on every first Page of each Documents in the training data. 

Then, for every input Document's Page, we determine if it's first or not by going through its Spans and comparing them 
to the set of Spans collected in the first step. If we have at least one Span of intersection between the current Page 
and the Spans from the first step, we believe it is the first Page.

Note that the more Documents we use in the "training" stage, the less intersecting Spans we are likely to find, so if at
the end of the tutorial you find that your first-Page Spans set is empty, try using a slice of the dataset instead of 
the whole like it is done in the example below. However, usually when ran on Documents within same Category, this 
algorithm should not return an empty set; if that is the case, you might want to check if your data is consistent (i.e. 
not in different languages, no occurrences of other Categories).

### Step-by-step explanation

In this section, we'll go through steps imitating initialization of `ContextAwareFileSplittingModel` class which you can 
find in the full code block in the lower part of this page. A class itself is already implemented and can be imported via
`from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel`.

Let's start with making all the necessary imports and initializing the class of `ContextAwareFileSplittingModel`:

```python
import json
import pathlib
import os
import shutil

from copy import deepcopy

from konfuzio_sdk.data import Page, Project
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.trainer.file_splitting import AbstractFileSplittingModel, SplittingAI
from konfuzio_sdk.trainer.information_extraction import load_model
from konfuzio_sdk.utils import get_timestamp


class ContextAwareFileSplittingModel(AbstractFileSplittingModel):

 def __init__(self, *args, **kwargs):

  """Initialize the ContextAwareFileSplittingModel."""
 super().__init__()
 self.name = self.__class__.__name__
 self.documents = None
 self.test_documents = None
 self.categories = None
 self.tokenizer = None
 self.first_page_strings = None
 self.path = None

```
The class inherits from `AbstractFileSplittingModel`, so we run `super().__init__()` for proper inheritance of the 
attributes. Attributes are set to `None` to be assigned explicitly upon usage later. `documents` and `test_documents` will be 
used for "training" (gathering unique first-page Spans from the Documents) and testing respectively; `categories` will 
define groups of Documents within which the "training" will take place; `first_page_spans` will be defined with the 
result of `fit()` method running; `tokenizer` will be used for going through the Document's text and separating it into 
Spans. We will use it to process training and testing Documents, as well as any Document that will undergo splitting. 
This is done to ensure that texts in all of the Documents are split using the same logic (particularly tokenization by 
separating on `\n` whitespaces by ConnectedTextTokenizer, which is used in the example in the end of the page) and it 
will be possible to find common Spans. `path` is a full path of the model-to-be-saved.

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

A first method to define will be `fit()`. It creates a dictionary with the unique first-page Spans gathered by Category.
We iterate through each Category's Document. For each Document, we create a deepcopy. Deepcopied object is identical to 
the one being copied, but it is not referencing the original one or pointing towards it, unlike when simple `copy`  or 
variable reassignment is used. Deepcopied Document does not have Annotations.

Then, we run Tokenizer on the deepcopied Document, which creates a set of Spans for it, and iterate through the 
Document's Pages.

If the Page is first, we append a set of its Span offset strings to `cur_first_page_spans`; if it is not first, a set of
its Span offset strings is appended to `cur_not_first_page_spans`. We use offset_springs and not Spans themselves 
because while a same string can occur on different Pages, it might not necessarily be in the same position (with same 
start_offset and end_offset) and thus would not be counted as similar when compared to the Spans of an input Document.

```python
def fit(self, *args, **kwargs) -> dict:
 first_page_spans = {}
 for category in self.categories:
  cur_first_page_spans = []
  cur_non_first_page_spans = []
  for doc in category.documents():
   doc = deepcopy(doc)
   doc.category = category
   doc = self.tokenizer.tokenize(doc)
   for page in doc.pages():
    if page.number == 1:
     cur_first_page_spans.append({span.offset_string for span in page.spans()})
    else:
     cur_non_first_page_spans.append({span.offset_string for span in page.spans()})
  if not cur_first_page_spans:
   cur_first_page_spans.append(set())
  true_first_page_spans = set.intersection(*cur_first_page_spans)
  if not cur_non_first_page_spans:
   cur_non_first_page_spans.append(set())
  true_not_first_page_spans = set.intersection(*cur_non_first_page_spans)
  true_first_page_spans = true_first_page_spans - true_not_first_page_spans
  first_page_spans[category.id_] = true_first_page_spans
 self.first_page_strings = first_page_spans
 return first_page_spans
```

Secondly, we define `save()` method. For saving, it is possible to choose one of the two options: to save 
`first_page_spans` as JSON (`save_json` has to be set to True) or to save a whole instance of the class as a pickle, 
using parent class's `save()` method (`save_json` has to be set to False). `include_konfuzio`, `max_ram` and 
`reduce_weight` are parameters passed into a parent `save()` method and can be left with their default values for this 
tutorial's scope.

```python
    def save(
        self,
        save_json=True,
        include_konfuzio=False,
        max_ram=None,
        reduce_weight: bool = False,
) -> str:
 if save_json:
  self.path = self.output_dir + f'/{get_timestamp()}_first_page_spans.json'
  pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
  with open(self.path, 'w+') as f:
   json.dump(self.first_page_strings, f)
 else:
  self.path = super().save(include_konfuzio=include_konfuzio, max_ram=max_ram, reduce_weight=reduce_weight)
 return self.path
```

Then we define `load_json()` method for loading previosuly saved `first_page_spans` from a chosen `model_path`:

```python
    def load_json(self, model_path=""):
 with open(model_path, 'r') as f:
  spans = json.load(f)
 # converting str category.id_ values to back int because JSON converts them to str
 spans = {int(k): v for k, v in spans.items()}
 self.first_page_strings = spans
```

Lastly, we define `predict()` method that uses resulting `first_page_spans` gathered from `fit()`. A Page is accepted as
 an input and its Span set is checked for containing first-page Spans for each of the Categories. If there has been at 
least one intersection, a Page is predicted to be first; if there's no such intersections, it's predicted non-first.

```python
 def predict(self, page: Page) -> Page:
 for category in self.categories:
  intersection = {span.offset_string for span in page.spans()}.intersection(
   self.first_page_strings[category.id_]
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
file_splitting_model.first_page_strings = file_splitting_model.fit()

# save the gathered Spans
file_splitting_model.output_dir = project.model_folder
file_splitting_model.save(save_json=True)

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
- `save` is optional, as one way of saving that is readable by Konfuzio Server (pickling) is already defined in the 
parent class. It is not necessary to override it, unless there is the need to define a particular different saving 
strategy (i.e. JSON, like it is done here). In such case, the new implementation should include calling `super()`, so
that the pickling saving option is not lost.

Full code:

```python
import json
import pathlib
import sys

from copy import deepcopy

from konfuzio_sdk.data import Document, Page
from konfuzio_sdk.trainer.information_extraction import load_model, BaseModel
from konfuzio_sdk.trainer.file_splitting import AbstractFileSplittingModel
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.utils import get_timestamp

class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    """Fallback definition of a File Splitting Model."""

    def __init__(self, *args, **kwargs):
        """Initialize the ContextAwareFileSplittingModel."""
        super().__init__()
        self.documents = None
        self.test_documents = None
        self.categories = None
        self.tokenizer = None
        self.first_page_spans = None

    def fit(self, *args, **kwargs) -> dict:
        """
        Gather the Spans unique for first Pages in a given stream of Documents.

        :return: Dictionary with unique first-page Span sets by Category ID.
        """
        first_page_spans = {}
        for category in self.categories:
            cur_first_page_spans = []
            cur_non_first_page_spans = []
            for doc in category.documents():
                doc = deepcopy(doc)
                doc.category = category
                doc = self.tokenizer.tokenize(doc)
                for page in doc.pages():
                    if page.number == 1:
                        cur_first_page_spans.append({span.offset_string for span in page.spans()})
                    else:
                        cur_non_first_page_spans.append({span.offset_string for span in page.spans()})
            if not cur_first_page_spans:
                cur_first_page_spans.append(set())
            true_first_page_spans = set.intersection(*cur_first_page_spans)
            if not cur_non_first_page_spans:
                cur_non_first_page_spans.append(set())
            true_not_first_page_spans = set.intersection(*cur_non_first_page_spans)
            true_first_page_spans = true_first_page_spans - true_not_first_page_spans
            first_page_spans[category.id_] = list(true_first_page_spans)
        self.first_page_spans = first_page_spans
        return first_page_spans

    def save(self, save_json=True, include_konfuzio=False) -> str:
        """
        Save the resulting set of first-page Spans by Category.

        :param save_json: Whether to save JSON of first_page_strings or a pickle of the whole class.
        :type save_json: bool
        :param include_konfuzio: Enables pickle serialization as a value, not as a reference (for more info, read
        https://github.com/cloudpipe/cloudpickle#overriding-pickles-serialization-mechanism-for-importable-constructs).
        :type include_konfuzio: bool
        """
        if save_json:
            path = self.output_dir + f'/{get_timestamp()}_first_page_spans.json'
            pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            with open(path, 'w+') as f:
                json.dump(self.first_page_spans, f)
        else:
            path = super().save(include_konfuzio=include_konfuzio)
        return path

    def load_json(self, model_path=""):
        """
        Load JSON with previously gathered first_page_strings.

        :param model_path: Path for the JSON.
        :type model_path: str
        """
        with open(model_path, 'r') as f:
            spans = json.load(f)
        # converting str category.id_ values to back int because JSON converts them to str
        spans = {int(k): v for k, v in spans.items()}
        self.first_page_spans = spans

    def predict(self, page: Page) -> Page:
        """
        Take a Page as an input and return 1 for a first Page and 0 for a non-first Page.

        :param page: A Page to receive first or non-first label.
        :type page: Page
        :return: A Page with or without is_first_page label.
        """
        for category in self.categories:
            intersection = {span.offset_string for span in page.spans()}.intersection(
                self.first_page_spans[category.id_]
            )
            if len(intersection) > 0:
                page.is_first_page = True
                break
        return page

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
file_splitting_model.first_page_spans = file_splitting_model.fit()

# save the gathered Spans
file_splitting_model.output_dir = project.model_folder
file_splitting_model.save(save_json=True)

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

# SplittingAI can be run in two modes: returning a list of sub-Documents as the result of the input Document
# splitting or returning a copy of the input Document with Pages predicted as first having an attribute
# "is_first_page". The flag "return_pages" has to be True for the latter; let's use it
new_document = splitting_ai.propose_split_documents(test_document, return_pages=True)
```