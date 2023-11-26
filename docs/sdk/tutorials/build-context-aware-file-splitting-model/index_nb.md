---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

## Develop and save a Context-Aware File Splitting AI

---

**Prerequisites:**

- Data Layer concepts of Konfuzio: Project, Category, Span, Document, Page
- AI concepts of Konfuzio: File Splitting

**Difficulty:** Hard

**Goal:** Guide the user through the steps of constructing of a Context-Aware File Splitting AI to explain better the logic behind it.

---

### Introduction

It's common for multi-paged files to not be perfectly organized, and in some cases, multiple independent Documents may be included in a single file. To ensure that these Documents are properly processed and separated, we will be discussing a method for identifying and splitting them into individual, independent Sub-Documents that does not require any ML-based approach.

![Multi-part Document](multi_file_document_example.png)

_Multi-file Document Example_

Konfuzio SDK offers two ways for separating Documents that may be included in a single file. One of them is training 
the instance of the Textual File Splitting Model for file splitting that would predict whether a Page is first or 
not and running the Splitting AI with it. Another approach is context-aware file splitting logic which is presented by Context Aware File Splitting Model. This approach involves analyzing the contents of each Page and identifying similarities to the first Pages of the Document. It will allow us to define splitting points and divide the Document into multiple Sub-Documents. It's important to note 
that this approach is only effective for Documents written in the same language and that the process must be repeated 
for each Category.

In this tutorial, we will walk you through the process of setting up the `ContextAwareFileSplittingModel` class, which 
can be found in the code block at the bottom of this page. This class is already implemented and can be imported using 
`from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel`.

#### Imports and initializing the class

Any custom File Splitting AI (derived from `AbstractFileSplittingModel` class) requires having the following 
methods implemented:

- `__init__` to initialize key variables required by the custom AI;
- `fit` to define architecture and training that the model undergoes, i.e. a certain NN architecture or a custom hardcoded logic;
- `predict` to define how the model classifies Pages as first or non-first. **NB:** the classification needs to be 
run on the Page level, not the Document level – the result of classification is reflected in `is_first_page` attribute 
value, which is unique to the Page class and is not present in Document class. Pages with `is_first_page = True` become 
splitting points, thus, each new Sub-Document has a Page predicted as first as its starting point.

To begin, we will make all the necessary imports and initialize the class:

```python editable=true slideshow={"slide_type": ""} tags=["remove-output", "skip-execution", "nbval-skip"] vscode={"languageId": "plaintext"}
from konfuzio_sdk.data import Page, Category
from konfuzio_sdk.trainer.file_splitting import AbstractFileSplittingModel

class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    def __init__(self, categories: List[Category], tokenizer, *args, **kwargs):
        super().__init__(categories=categories)
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

#### ConnectedTextTokenizer explained

Here is an example of how ConnectedTextTokenizer works. At first, we have a Document with the untokenized text:

```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
import logging
from konfuzio_sdk.samples import LocalTextProject
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
logging.getLogger("konfuzio_sdk").setLevel(logging.ERROR)

project = LocalTextProject()
tokenizer = ConnectedTextTokenizer()
YOUR_DOCUMENT_ID = 9
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)
assert (
    test_document.text == "Hi all,\nI like bread.\n\fI hope to get everything done soon.\n\fMorning,\n\fI'm glad "
    "to see you.\n\fMorning,"
)
assert test_document.spans() == []
```

```python editable=true slideshow={"slide_type": ""}
# before tokenization
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)
print(test_document.text)
```

If we print this Document's Spans, we will see there are none.
```python
test_document.spans()
```

Let's tokenize the Document and check the Spans after that:
```python
test_document = tokenizer.tokenize(test_document)

test_document.spans()
```

#### Creating necessary methods of the class

The next method to define will be the `fit()` method. For each Category, we call `exclusive_first_page_strings` method, 
which allows us to gather the strings that appear on the first Page of each Document. `allow_empty_categories` allows 
for returning empty lists for Categories that haven't had any exclusive first-page strings found across their Documents. 
This means that those Categories would not be used in the prediction process.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
    def fit(self, allow_empty_categories: bool = False, *args, **kwargs):
        for category in self.categories:
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

Then, we define `predict()` method. The method accepts a Page as an input and checks its Span set for containing 
first-page strings for each of the Categories. If there is at least one intersection, the Page is predicted to be a 
first Page. If there are no intersections, the Page is predicted to be a non-first Page.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
    def predict(self, page: Page) -> Page:
        self.check_is_ready()
        page.is_first_page = False
        for category in self.categories:
            cur_first_page_strings = category.exclusive_first_page_strings(tokenizer=self.tokenizer)
            intersection = {span.offset_string.strip('\f').strip('\n') for span in page.spans()}.intersection(
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

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
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

### Conclusion
In this tutorial, we have walked through the essential steps for constructing the Context-Aware File Splitting Model. Below is the full code of the class:

```python editable=true slideshow={"slide_type": ""} tags=["remove-output"] vscode={"languageId": "plaintext"}
import logging
from typing import List
from konfuzio_sdk.data import Page, Category
from konfuzio_sdk.trainer.file_splitting import AbstractFileSplittingModel

logger = logging.getLogger(__name__)

class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    def __init__(self, categories: List[Category], tokenizer, *args, **kwargs):
        super().__init__(categories=categories)
        self.output_dir = self.project.model_folder
        self.tokenizer = tokenizer
        self.requires_text = True
        self.requires_images = False

    def fit(self, allow_empty_categories: bool = False, *args, **kwargs):
        for category in self.categories:
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
        self.check_is_ready()
        page.is_first_page = False
        for category in self.categories:
            cur_first_page_strings = category.exclusive_first_page_strings(tokenizer=self.tokenizer)
            intersection = {span.offset_string.strip('\f').strip('\n') for span in page.spans()}.intersection(
                cur_first_page_strings
            )
            if len(intersection) > 0:
                page.is_first_page = True
                break
        page.is_first_page_confidence = 1
        return page

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

### What's next?

- [Learn how to train and use Context-Aware File Splitting Model](https://dev.konfuzio.com/sdk/tutorials/tutorials/build-context-aware-file-splitting-model/index.html)
- [Get to know how to build a custom File Splitting AI](https://dev.konfuzio.com/sdk/tutorials/tutorials/file-splitting-evaluation/index.html)

