## Splitting for multi-file Documents: Step-by-step guide

### Intro

Not all multipage files that we process are always neatly scanned and organized. Sometimes we can have more than one actual 
Document in a stream pf pages; such files need to be properly processed and split into several independent Documents. 

![multi-file Document example](https://miro.medium.com/max/4800/1*P6BghoNH9LglgNV3SiDTVg.webp)
_Multi-file Document Example_ by [Qaisar Tanvir](https://medium.com/@qaisartanvir)

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
find in the full code block in the lower part of this page. 

Let's start with making all the necessary imports and initializing the class of `ContextAwareFileSplittingModel`:
```python
import pickle

from copy import deepcopy

from konfuzio_sdk.data import Page, Project
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.trainer.file_splitting import AbstractFileSplittingModel

class ContextAwareFileSplittingModel(AbstractFileSplittingModel):

    def __init__(self, *args, **kwargs):
        self.train_data = None 
        self.test_data = None
        self.categories = None
        self.first_page_spans = None
        self.tokenizer = None
        
```
The class inherits from `AbstractFileSplittingModel`. Attributes are set to `None` to be assigned explicitly upon usage 
later. `train_data` and `test_data` will be used for "training" (gathering unique first-page Spans from the Documents) 
and testing respectively; `categories` will define groups of Documents within which the "training" will take place;
`first_page_spans` will be defined with the result of `fit()` method running; `tokenizer` will be used for going through 
the Document's text and separating it into Spans. We will use it to process training and testing Documents, as well as 
any Document that will undergo splitting. This is done to ensure that texts in all of the Documents are split using the 
same logic (particularly tokenization by separating on `\n` whitespaces by ConnectedTextTokenizer, which is used in the 
example in the end of the page) and it will be possible to find common Spans. 

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
                doc = self.tokenizer.tokenize(deepcopy(doc))
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
        self.first_page_spans = first_page_spans
        return first_page_spans
```

Secondly, we define `save()` method which is basically pickling the output of the `fit()` method.
```python
def save(self, model_path=""):
    with open(model_path + '/first_page_spans.pickle', "wb") as pickler:
        pickle.dump(self.first_page_spans, pickler)
```

Lastly, we define `predict()` method that uses resulting `first_page_spans` gathered from `fit()`. A Page is accepted as
 an input and its Span set is checked for containing first-page Spans for each of the Categories. If there has been at 
least one intersection, a Page is predicted to be first; if there's no such intersections, it's predicted non-first.
```python
def predict(self, page: Page) -> int:
        intersections = {}
        for category in self.categories:
            intersection = len(
                {span.offset_string for span in page.spans()}.intersection(self.first_page_spans[category.id_])
            )
            if intersection > 0:
                intersections[category.id_] = intersection
        if len(intersections) > 0:
            return 1
        else:
            return 0
```

A quick example of the class's usage:
```python
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
file_splitting_model.first_page_spans = file_splitting_model.fit()

# run the prediction
for page in test_document.pages():
    pred = file_splitting_model.predict(page)
    if pred == 1:
        print('Page {} is predicted as the first.'.format(page.number))
    else:
        print('Page {} is predicted as the non-first.'.format(page.number))
```

Full code:

```python
import pickle

from copy import deepcopy

from konfuzio_sdk.data import Page, Project
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.trainer.file_splitting import AbstractFileSplittingModel

class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    """Fallback definition of a File Splitting Model."""

    def __init__(self, *args, **kwargs):
        """Initialize the ContextAwareFileSplittingModel."""
        self.train_data = None 
        self.test_data = None
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
                doc = self.tokenizer.tokenize(deepcopy(doc))
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
        self.first_page_spans = first_page_spans
        return first_page_spans

    def save(self, model_path=""):
        """
        Save the resulting set of first-page Spans by Category.
        :param model_path: Path to save the set to.
        :type model_path: str
        """
        with open(model_path + '/first_page_spans.pickle', "wb") as pickler:
            pickle.dump(self.first_page_spans, pickler)

    def predict(self, page: Page) -> int:
        """
        Take a Page as an input and return 1 for a first Page and 0 for a non-first Page.
        :param page: A Page to receive first or non-first label.
        :type page: Page:
        :return: A label of a first or a non-first Page.
        """
        intersections = {}
        for category in self.categories:
            intersection = len(
                {span.offset_string for span in page.spans()}.intersection(self.first_page_spans[category.id_])
            )
            if intersection > 0:
                intersections[category.id_] = intersection
        if len(intersections) > 0:
            return 1
        else:
            return 0

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
file_splitting_model.first_page_spans = file_splitting_model.fit()

# run the prediction
for page in test_document.pages():
    pred = file_splitting_model.predict(page)
    if pred == 1:
        print('Page {} is predicted as the first.'.format(page.number))
    else:
        print('Page {} is predicted as the non-first.'.format(page.number))
```