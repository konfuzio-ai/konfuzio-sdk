## Splitting for multi-Document files: Step-by-step guide

### Intro

It's common for multipage files to not be perfectly organized, and in some cases, multiple independent Documents may be 
included in a single file. To ensure that these Documents are properly processed and separated, we will be discussing a 
method for identifying and splitting them into individual, independent Sub-documents.

.. image:: /sdk/examples/file-splitting-class/multi_file_document_example.png

_Multi-file Document Example_

Konfuzio SDK offers two ways for separating Documents that may be included in a single file. One of them is training 
the instance of the Multimodal File Splitting Model for file splitting that would predict whether a Page is first or 
not and running the Splitting AI with it. Multimodal File Splitting Model is a combined approach based on architecture 
that processes textual and visual data from the Documents separately (in our case, using BERT and VGG19 simplified 
architectures respectively) and then combines the outputs which go into a Multi-layer Perceptron architecture as 
inputs. A more detailed scheme of the architecture can be found further.

If you hover over the image you can zoom or use the full page mode.

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers tags lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;url&quot;:&quot;https://raw.githubusercontent.com/konfuzio-ai/konfuzio-sdk/master/docs/sdk/examples/file-splitting-class/fusion_model.drawio&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/embed2.js?&fetch=https%3A%2F%2Fraw.githubusercontent.com%2Fkonfuzio-ai%2Fkonfuzio-sdk%2Fmaster%2Fdocs%2Fsdk%2Fexamples%2Ffile-splitting-class%2Ffusion_model.drawio"></script>

Another approach is context-aware file splitting logic which is presented by Context Aware File Splitting Model. This 
approach involves analyzing the contents of each Page and identifying similarities to the first Pages of the Document. 
It will allow us to define splitting points and divide the Document into multiple Sub-documents. It's important to note 
that this approach is only effective for Documents written in the same language and that the process must be repeated 
for each Category. In this tutorial, we will explain how to implement the class for this model step by step.

If you are unfamiliar with the SDK's main concepts (like Page or Span), you can get to know them at [Data Layer Concepts](https://dev.konfuzio.com/sdk/explanations.html#data-layer-concepts).


### Quick explanation

The first step in implementing this method is "training": this involves tokenizing the Document by splitting its text 
into parts, specifically into strings without line breaks. We then gather the exclusive strings from Spans, which are 
the parts of the text in the Page, and compare them to the first Pages of each Document in the training data.

Once we have identified these strings, we can use them to determine whether a Page in an input Document is a first Page 
or not. We do this by going through the strings in the Page and comparing them to the set of strings collected in the 
training stage. If we find at least one string that intersects between the current Page and the strings from the first 
step, we believe it is the first Page.

Note that the more Documents we use in the training stage, the less intersecting strings we are likely to find. If you 
find that your set of first-page strings is empty, try using a smaller slice of the dataset instead of the whole set. 
Generally, when used on Documents within the same Category, this algorithm should not return an empty set. If that is 
the case, it's worth checking if your data is consistent, for example, not in different languages or containing other 
Categories.

### Step-by-step explanation

In this section, we will walk you through the process of setting up the `ContextAwareFileSplittingModel` class, which 
can be found in the code block at the bottom of this page. This class is already implemented and can be imported using 
`from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel`.

Note that any custom File Splitting AI (derived from `AbstractFileSplittingModel` class) requires having the following 
methods implemented:
- `__init__` to initialize key variables required by the custom AI;
- `fit` to define architecture and training that the model undergoes, i.e. a certain NN architecture or a custom 
- hardcoded logic;
- `predict` to define how the model classifies Pages as first or non-first. **NB:** the classification needs to be 
run on the Page level, not the Document level – the result of classification is reflected in `is_first_page` attribute 
value, which is unique to the Page class and is not present in Document class. Pages with `is_first_page = True` become 
splitting points, thus, each new Sub-Document has a Page predicted as first as its starting point.

To begin, we will make all the necessary imports and initialize the `ContextAwareFileSplittingModel` class:
```python
import logging

from typing import List

from konfuzio_sdk.data import Page, Category
from konfuzio_sdk.trainer.file_splitting import AbstractFileSplittingModel
from konfuzio_sdk.trainer.information_extraction import load_model
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer

class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    def __init__(self, categories: List[Category], tokenizer, *args, **kwargs):
        super().__init__(categories=categories)
        self.name = self.__class__.__name__
        self.output_dir = self.project.model_folder
        self.tokenizer = tokenizer
        self.requires_text = True
        self.requires_images = False
```
The class inherits from `AbstractFileSplittingModel`, so we run `super().__init__(categories=categories)` to properly 
inherit its attributes. The `tokenizer` attribute will be used to process the text within the Document, separating it 
into Spans. This is done to ensure that the text in all the Documents is split using the same logic (particularly 
tokenization by separating on `\n` whitespaces by ConnectedTextTokenizer, which is used in the example in the end of the 
page) and it will be possible to find common Spans. It will be used for training and testing Documents as well as any 
Document that will undergo splitting. It's important to note that if you run fitting with one Tokenizer and then 
reassign it within the same instance of the model, all previously gathered strings will be deleted and replaced by new 
ones. `requires_images` and `requires_text` determine whether these types of data are used for prediction; this is 
needed for distinguishing between preprocessing types once a model is passed into the Splitting AI.   

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

The first method to define will be the `fit()` method. For each Category, we call `exclusive_first_page_strings` method, 
which allows us to gather the strings that appear on the first Page of each Document. `allow_empty_categories` allows 
for returning empty lists for Categories that haven't had any exclusive first-page strings found across their Documents. 
This means that those Categories would not be used in the prediction process.
```python
    def fit(self, allow_empty_categories: bool = False, *args, **kwargs):
        for category in self.categories:
            # method exclusive_first_page_strings fetches a set of first-page strings exclusive among the Documents
            # of a given Category. they can be found in _exclusive_first_page_strings attribute of a Category after
            # the method has been run. this is needed so that the information remains even if local variable
            # cur_first_page_strings is lost.
            cur_first_page_strings = category.exclusive_first_page_strings(tokenizer=self.tokenizer)
            if not cur_first_page_strings:
                if allow_empty_categories:
                    logger.warning(
                        f'No exclusive first-page strings were found for {category}, so it will not be used '
                        f'at prediction.'
                    )
                else:
                    raise ValueError(f'No exclusive first-page strings were found for {category}.')
```

Next, we define `predict()` method. The method accepts a Page as an input and checks its Span set for containing 
first-page strings for each of the Categories. If there is at least one intersection, the Page is predicted to be a 
first Page. If there are no intersections, the Page is predicted to be a non-first Page.
```python
    def predict(self, page: Page) -> Page:
        for category in self.categories:
            # exclusive_first_page_strings calls an implicit _exclusive_first_page_strings attribute once it was 
            # already calculated during fit() method so it is not a recurrent calculation each time.
            if not category.exclusive_first_page_strings:
                raise ValueError(f"Cannot run prediction as {category} does not have exclusive_first_page_strings.")
        page.is_first_page = False
        for category in self.categories:
            cur_first_page_strings = category.exclusive_first_page_strings(tokenizer=self.tokenizer)
            intersection = {span.offset_string for span in page.spans()}.intersection(
                cur_first_page_strings
            )
            if len(intersection) > 0:
                page.is_first_page = True
                break
        page.is_first_page_confidence = 1
        return page
```

Lastly, a `check_is_ready()` method is defined. This method is used to ensure that a model is ready for prediction: the
checks cover that the Tokenizer and a set of Categories is defined, and that at least one of the Categories has 
exclusive first-page strings.
```python
    def check_is_ready(self):
        if self.tokenizer is None:
            raise AttributeError(f'{self} missing Tokenizer.')

        if not self.categories:
            raise AttributeError(f'{self} requires Categories.')

        empty_first_page_strings = [
            category
            for category in self.categories
            if not category.exclusive_first_page_strings(tokenizer=self.tokenizer)
        ]
        if len(empty_first_page_strings) == len(self.categories):
            raise ValueError(
                f"Cannot run prediction as none of the Categories in {self.project} have "
                f"_exclusive_first_page_strings."
            )
```

A quick example of the class's usage:

```python
# initialize a Project and fetch a test Document of your choice
project = Project(id_=YOUR_PROJECT_ID)
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)

# initialize a Context Aware File Splitting Model and fit it

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

# usage with the Splitting AI – you can load a pre-saved model or pass an initialized instance as the input
# in this example, we load a previously saved one
model = load_model(project.model_folder)

# initialize the Splitting AI
splitting_ai = SplittingAI(model)

# Splitting AI is a more high-level interface to the Context Aware File Splitting Model and any other models that can be 
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
```

Full code:

```python
import logging

from typing import List

from konfuzio_sdk.data import Page, Category, Project
from konfuzio_sdk.trainer.file_splitting import AbstractFileSplittingModel, SplittingAI
from konfuzio_sdk.trainer.information_extraction import load_model
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer

logger = logging.getLogger(__name__)

class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    """
    A File Splitting Model that uses a context-aware logic.

    Context-aware logic implies a rule-based approach that looks for common strings between the first Pages of all
    Category's Documents.
    """

    def __init__(self, categories: List[Category], tokenizer, *args, **kwargs):
        """
        Initialize the Context Aware File Splitting Model.

        :param categories: A list of Categories to run training/prediction of the model on.
        :type categories: List[Category]
        :param tokenizer: Tokenizer used for processing Documents on fitting when searching for exclusive first-page
        strings.
        :raises ValueError: When an empty list of Categories is passed into categories argument.
        :raises ValueError: When a list passed into categories contains elements other than Categories.
        :raises ValueError: When a list passed into categories contains at least one Category with no Documents or test
        Documents.
        """
        super().__init__(categories=categories)
        self.output_dir = self.project.model_folder
        self.tokenizer = tokenizer
        self.requires_text = True
        self.requires_images = False

    def fit(self, allow_empty_categories: bool = False, *args, **kwargs):
        """
        Gather the strings exclusive for first Pages in a given stream of Documents.

        Exclusive means that each of these strings appear only on first Pages of Documents within a Category.

        :param allow_empty_categories: To allow returning empty list for a Category if no exclusive first-page strings
        were found during fitting (which means prediction would be impossible for a Category).
        :type allow_empty_categories: bool
        :raises ValueError: When allow_empty_categories is False and no exclusive first-page strings were found for
        at least one Category.
        """
        for category in self.categories:
            # method exclusive_first_page_strings fetches a set of first-page strings exclusive among the Documents
            # of a given Category. they can be found in _exclusive_first_page_strings attribute of a Category after
            # the method has been run. this is needed so that the information remains even if local variable
            # cur_first_page_strings is lost.
            cur_first_page_strings = category.exclusive_first_page_strings(tokenizer=self.tokenizer)
            if not cur_first_page_strings:
                if allow_empty_categories:
                    logger.warning(
                        f'No exclusive first-page strings were found for {category}, so it will not be used '
                        f'at prediction.'
                    )
                else:
                    raise ValueError(f'No exclusive first-page strings were found for {category}.')

    def predict(self, page: Page) -> Page:
        """
        Predict a Page as first or non-first.

        :param page: A Page to receive first or non-first label.
        :type page: Page
        :return: A Page with a newly predicted is_first_page attribute.
        """
        self.check_is_ready()
        page.is_first_page = False
        for category in self.categories:
            cur_first_page_strings = category.exclusive_first_page_strings(tokenizer=self.tokenizer)
            intersection = {span.offset_string for span in page.spans()}.intersection(
                cur_first_page_strings
            )
            if len(intersection) > 0:
                page.is_first_page = True
                break
        page.is_first_page_confidence = 1
        return page

    def check_is_ready(self):
        """
        Check File Splitting Model is ready for inference.

        :raises AttributeError: When no Tokenizer or no Categories were passed.
        :raises ValueError: When no Categories have _exclusive_first_page_strings.
        """
        if self.tokenizer is None:
            raise AttributeError(f'{self} missing Tokenizer.')

        if not self.categories:
            raise AttributeError(f'{self} requires Categories.')

        empty_first_page_strings = [
            category
            for category in self.categories
            if not category.exclusive_first_page_strings(tokenizer=self.tokenizer)
        ]
        if len(empty_first_page_strings) == len(self.categories):
            raise ValueError(
                f"Cannot run prediction as none of the Categories in {self.project} have "
                f"_exclusive_first_page_strings."
            )

# initialize a Project and fetch a test Document of your choice
project = Project(id_=YOUR_PROJECT_ID)
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)

# initialize a Context Aware File Splitting Model and fit it

file_splitting_model = ContextAwareFileSplittingModel(categories=project.categories, tokenizer=ConnectedTextTokenizer())
file_splitting_model.fit()

# save the model
file_splitting_model.save()

# run the prediction
for page in test_document.pages():
 pred = file_splitting_model.predict(page)
 if pred.is_first_page:
  print('Page {} is predicted as the first.'.format(page.number))
 else:
  print('Page {} is predicted as the non-first.'.format(page.number))

# usage with the Splitting AI – you can load a pre-saved model or pass an initialized instance as the input
# in this example, we load a previously saved one
model = load_model(project.model_folder)

# initialize the Splitting AI
splitting_ai = SplittingAI(model)

# Splitting AI is a more high-level interface to Context Aware File Splitting Model and any other models that can be 
# developed for file-splitting purposes. It takes a Document as an input, rather than individual Pages, because it 
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
```