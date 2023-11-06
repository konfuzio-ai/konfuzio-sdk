---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: konfuzio
    language: python
    name: python3
---

## Tokenization

---

**Prerequisites:**
- Install the Konfuzio SDK.
- Have access to a Project on the Konfuzio platform.
- Be familiar with the following concepts:
    - [Document](ADD-LINK)
    - [Bounding Box](ADD-LINK)
    - [Spans](ADD-LINK)
    - [Label](ADD-LINK)

**Difficulty:** This tutorial is suitable for beginners in NLP and the Konfuzio SDK.

**Goal:** Be familiar with the concept of tokenization and master how different tokenization approaches can be used with Konfuzio.

---

### Introduction
In this tutorial, we will explore the concept of tokenization and the various tokenization strategies available in the Konfuzio SDK. Tokenization is a foundational tool in natural language processing (NLP) that involves breaking text into smaller units called tokens. We will focus on the `WhitespaceTokenizer`, `Label-Specific Regex Tokenizer`, `ParagraphTokenizer`, and `SentenceTokenizer` as different tools for different tokenization tasks.



### Whitespace Tokenization
The `WhitespaceTokenizer`, part of the Konfuzio SDK, is a simple yet effective tool for basic tokenization tasks. It segments text into tokens using white spaces as natural delimiters.

#### Use case: Retrieving the Word Bounding Box for a Document
In this section, we will walk through how to use the `WhitespaceTokenizer` to extract word-level [Bounding Boxes](ADD-LINK) for a Document.

We will use the Konfuzio SDK to tokenize the Document and identify word-level [Spans](ADD-LINK), which can then be visualized or used to extract bounding box information.


##### Steps
1. Import necessary modules

```python tags=["remove-cell"]
# This is necessary to make sure we can import from 'tests'
import sys
sys.path.insert(0, '../../../../')
```

```python tags=["remove-cell"]
from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID, TEST_PAYSLIPS_CATEGORY_ID, TEST_CATEGORIZATION_DOCUMENT_ID
```

```python
from copy import deepcopy
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer
```

2. Initialize a Project and a Document instance. The variables `TEST_PROJECT_ID` and `TEST_DOCUMENT_ID` are special variables used at Konfuzio for internal testing. Make sure to use a Project and Document id to which you have access.

```python tags=["remove-output"]
project = Project(id_=TEST_PROJECT_ID)
document = project.get_document_by_id(TEST_DOCUMENT_ID)

# We create a copy of the document object to make sure it contains no Annotations
document = deepcopy(document)

```

3. Tokenize the Document
This process involves splitting the Document into word-level Spans using the WhitespaceTokenizer.

```python tags=["remove-output"]
tokenizer = WhitespaceTokenizer()
tokenized_spans = tokenizer.tokenize(document)
```

4. Visualize word-level Annotations
We now visually check that the Bounding Boxes are correctly assigned.

```python
document.get_page_by_index(0).get_annotations_image(display_all=True)
```

Observe how each individual word is enclosed in a Bounding Box. Also note that these Bounding Boxes have no [Label](ADD-LINK) associated, thereby the placeholder 'NO_LABEL' is shown above each Bounding Box.


5. Retrieving Bounding Boxes

Each Bounding Box corresponds is associated to a specific word and is defined by four coordinates:
- x0 and y0 specify the coordinates of the bottom left corner;
- x1 and y1 specify the coordinates of the top right corner

Which allow to determine the size and position of the Box on the page.

All Bounding Boxes calculated after tokenization occurred can be obtained as follows:

```python
span_bboxes = [span.bbox() for span in document.spans()]
```

Let us inspect the first 10 Bounding Boxes' coordinates to verify that each comprises 4 coordinate points.

```python
span_bboxes[:10]
```

To summarize, here is the full code:

```python tags=["skip-execution", "nbval-skip"]
from copy import deepcopy
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer

project = Project(id_=TEST_PROJECT_ID)
document = project.get_document_by_id(TEST_DOCUMENT_ID)

# We create a copy of the document object to make sure it contains no Annotations
document = deepcopy(document)

# Initialize tokenizer
tokenizer = WhitespaceTokenizer()
tokenized_spans = tokenizer.tokenize(document)

# Retrieve Bounding Boxes
span_bboxes = [span.bbox() for span in document.spans()]

document.get_page_by_index(0).get_annotations_image(display_all=True)
```

Note: remember to use a Project and Document id to which you have access. The variables TEST_PROJECT_ID and TEST_DOCUMENT_ID are special variables used at Konfuzio for internal testing.


### Regex Tokenization for Specific Labels
In some cases, especially when dealing with intricate annotation strings, a custom Regex Tokenizer can offer a powerful solution. Unlike basic `WhitespaceTokenizers`, which split text based on spaces, a Regex Tokenizer utilizes regular expressions to define complex patterns for identifying and extracting tokens. This tutorial will guide you through the process of creating and training your own Regex Tokenizer, providing you with a versatile tool to handle even the most challenging tokenization tasks. Let's get started!

In this example, we will see how to find regular expressions that match with occurrences of the "Lohnart" (which approximately means _type of salary_ in German) Label in the training data, namely Documents that have been annotated by hand.


1. Import modules

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import RegexTokenizer
from konfuzio_sdk.tokenizer.base import ListTokenizer
```

2. Initialize the Project and obtain the Category. We need to obtain an instance of the Category which we will be using at a later point exactly to achieve what we need: find the best regular expression matching our Label of interest.

```python tags=["remove-output"]
my_project = Project(id_=TEST_PROJECT_ID)
category = my_project.get_category_by_id(id_=TEST_PAYSLIPS_CATEGORY_ID)
```

3. We use the ListTokenizer, a class that provides a way to organize and apply multiple tokenizers to a Document, allowing for complex tokenization pipelines in natural language processing tasks. In this case, we simply use to hold our regular expression tokenizers.

```python
tokenizer_list = ListTokenizer(tokenizers=[])
```

4. Retrieve the "Lohnart" Label using its name. 
Note that if you have the Label ID, you could also use the `Project.get_label_by_id` method.

```python
label = my_project.get_label_by_name("Lohnart")
```

5. Find Regular Expressions and Create RegexTokenizers. We now use `Label.find_regex` to algorithmically search for the best fitting regular expressions matching the Annotations associated with this Label. Each RegexTokenizer is collected in the container object `tokenizer_list`.

```python
rgxs = label.find_regex(category=category)
len(rgxs)
```

```python
for regex in rgxs:
    print(regex)  # to show how the regex can look, for instance: (?:(?P<Label_861_N_672673_1638>\d\d\d\d))[ ]{1,2}
    regex_tokenizer = RegexTokenizer(regex=regex)
    tokenizer_list.tokenizers.append(regex_tokenizer)
```

6. Use the new Tokenizer to Create New Annotations. Finally, we can use the TokenizerList instance to create new `NO_LABEL` Annotations for each string in the Document matching the regex patterns found.

```python tags=["remove-output"]
# You can then use it to create an Annotation 
document = my_project.get_document_by_id(TEST_DOCUMENT_ID)
document = tokenizer_list.tokenize(document)
```

To summarize, here is the complete code of our RegEx Tokenization example

```python tags=["skip-execution", "nbval-skip"]
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import RegexTokenizer
from konfuzio_sdk.tokenizer.base import ListTokenizer

my_project = Project(id_=TEST_PROJECT_ID)
category = my_project.get_category_by_id(id_=TEST_PAYSLIPS_CATEGORY_ID)

tokenizer_list = ListTokenizer(tokenizers=[])
label = my_project.get_label_by_name("Lohnart")
rgxs = label.find_regex(category=category)

for regex in rgxs:
    print(regex)  # to show how the regex can look, for instance: (?:(?P<Label_861_N_672673_1638>\d\d\d\d))[ ]{1,2}
    regex_tokenizer = RegexTokenizer(regex=regex)
    tokenizer_list.tokenizers.append(regex_tokenizer)

# You can then use it to create an Annotation 
document = my_project.get_document_by_id(TEST_DOCUMENT_ID)
document = tokenizer_list.tokenize(document)

```

### Paragraph Tokenization


The `ParagraphTokenizer` class is a specialized tool designed to split a Document into meaningful sections, creating Annotations. It offers two modes of operation: `detectron` and `line_distance`.

To determine the mode of operation, the `mode` constructor parameter is used, it can take two values: `detectron` (default) or `line_distance`. In `detectron` mode, the Tokenizer employs a fine-tuned Detectron2 model to assist in Document segmentation. While this mode tends to be more accurate, it is slower as it requires making an API call to the model hosted on Konfuzio servers. On the other hand, the `line_distance` mode uses a rule-based approach that is faster but less accurate, especially with Documents having two columns or other complex layouts.



#### line_distance Approach
It provides an efficient way to segment Documents based on line heights, making it particularly useful for simple, single-column formats. Although it may have limitations with complex layouts, its swift processing and relatively accurate results make it a practical choice for tasks where speed is a priority and the Document structure isn't overly complicated.

##### Parameters
The behavior of the `line_distance` approach can be adjusted with the following parameters.
- `line_height_ratio`: (Float) Specifies the ratio of the median line height used as a threshold to create a new paragraph when using the Tokenizer in `line_distance` mode. The default value is 0.8. If you find that the Tokenizer is not creating new paragraphs when it should, you can try **lowering** this value. Alternatively, if the Tokenizer is creating too many paragraphs, you can try **increasing** this value.

- `height`: (Float) This optional parameter allows you to define a specific line height threshold for creating new paragraphs. If set to None, the Tokenizer uses an intelligently calculated height threshold.

Using the `ParagraphTokenizer` in `line_distance` mode boils down to creating a `ParagraphTokenizer` instance and use it do tokenize a Document. As we did before we fetch the image representing the first page of the Document to visualize the Annotations generated by the tokenizer.

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.paragraph_and_sentence import ParagraphTokenizer

# Initialize a Project
project = Project(id_=TEST_PROJECT_ID)

# Fetch Document
document = project.get_document_by_id(TEST_DOCUMENT_ID)

# Create the ParagraphTokenizer
tokenizer = ParagraphTokenizer(mode='line_distance')

# Tokenize the Document
document = tokenizer(document)

# Fetch first page as image
document.get_page_by_index(0).get_annotations_image(display_all=True)  # display_all to show NO_LABEL Annotations
```

Due to the complexity of the document we processed, the `line_distance` approach does not perform very well. We can see that a simpler document that has a more linear structure gives better results:

```python
from tests.variables import TEST_PROJECT_ID
from konfuzio_sdk.data import Project, Document
from konfuzio_sdk.tokenizer.paragraph_and_sentence import ParagraphTokenizer
from copy import deepcopy

# Initialize a Project
project = Project(id_=TEST_PROJECT_ID)

# Prepare Document
doc = deepcopy(Document.from_file("heroic.pdf", project=project, sync=True))

# Create the ParagraphTokenizer
tokenizer = ParagraphTokenizer(mode='line_distance')

# Tokenize the Document
_ = tokenizer(doc)

# Fetch first page as image
doc.get_page_by_index(0).get_annotations_image(display_all=True)  # display_all to show NO_LABEL Annotations
```

```python tags=["remove-cell"]
doc.delete(delete_online=True)
```

Note that this example uses a PDF file named 'heroic.pdf', you can download this file [here](ADD-LINK) in case you wish to try running this code.


#### Detectron (CV) Approach
With the Computer Vision (CV) approach, we can create Labels, identify figures, tables, lists, texts, and titles, thereby giving us a comprehensive understanding of the Document's structure.

Using the Computer Vision approach might require more processing power and might be slower compared to the `line_distance` approach, but the significant leap in the comprehensiveness of the output makes it a powerful tool.

##### Parameters
- `create_detectron_labels`: (Boolean, default: False) if set to True, Labels will be created and assigned to the document. Labels may include `figure`, `table`, `list`, `text` and `title`. If this option is set to False, the Tokenizer will create `NO_LABEL` Annotations.


Using a Tokenizer in `detectron` mode boils down to passing `detectron` ot the `mode` argument of the `ParagraphTokenizer`:

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.paragraph_and_sentence import ParagraphTokenizer

# initialize a Project and fetch a Document to tokenize
project = Project(id_=TEST_PROJECT_ID)

document = project.get_document_by_id(TEST_DOCUMENT_ID)

tokenizer = ParagraphTokenizer(mode='detectron', create_detectron_labels=True)

_ = tokenizer(document)

document.get_page_by_index(0).get_annotations_image()
```

Comparing this result to tokenizing the same document with the `line_distance` approach it is evident that the `detectron` mode can extract a meaningful structure from a relatively complex document.


### Sentence Tokenization


The `SentenceTokenizer` is a specialized tokenizer designed to split text into sentences. Similarly to the ParagraphTokenizer, it has two modes of operation: `detectron` and `line_distance`, and accepts the same additional parameters: `line_height_ratio`, `height` and `create_detectron_labels`.

Using it is straightforward and comparable to using the `ParagraphTokenizer`:

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.paragraph_and_sentence import SentenceTokenizer
from copy import deepcopy

# Initialize a Project
project = Project(id_=TEST_PROJECT_ID, update=True)

# Prepare Document
doc = project.get_document_by_id(5679477)

# Create the SentenceTokenizer
tokenizer = SentenceTokenizer(mode='detectron')

# Tokenize the Document
_ = tokenizer(doc)
```

```python
# Fetch first page as image
doc.get_page_by_index(1).get_annotations_image(display_all=True)
```

### Choosing the Right Tokenizer

When it comes to Natural Language Processing (NLP), choosing the correct Tokenizer can make a significant impact on your system's performance and accuracy. The Konfuzio SDK offers several tokenization options, each suited to different tasks:

- **WhitespaceTokenizer**: Perfect for basic word-level processing. This Tokenizer breaks text into chunks separated by white spaces. It is ideal for straightforward tasks such as basic keyword extraction.

- **Label-Specific Regex Tokenizer**: Known as “Character” detection mode on the Konfuzio platform, this Tokenizer offers more specialized functionality. It uses Annotations of a Label within a training set to pinpoint and tokenize precise chunks of text. It’s especially effective for tasks like entity recognition, where accuracy is paramount. By recognizing specific word or character patterns, it allows for more precise and nuanced data processing.

- **ParagraphTokenizer**: Identifies and separates larger text chunks - paragraphs. This is beneficial when your text’s interpretation relies heavily on the context at the paragraph level.

- **SentenceTokenizer**: Segments text into sentences. This is useful when the meaning of your text depends on the context provided at the sentence level.

Choosing the right Tokenizer is a matter of understanding your NLP task, the structure of your data, and the degree of detail your processing requires. By aligning these elements with the functionalities provided by the different Tokenizers in the Konfuzio SDK, you can select the best tool for your task.


### Verify That a Tokenizer Finds All Labels

To help you choose the right Tokenizer for your task, it can be useful to try out different Tokenizers and see which Spans are found by which Tokenizer. The `Label` class provides a method called `spans_not_found_by_tokenizer` that can he helpful in this regard.

Here is an example of how to use the `Label.spans_not_found_by_tokenizer` method. This will allow you to determine if a RegexTokenizer is suitable at finding the Spans of a Label, or what Spans might have been annotated wrong. Say, you have a number of Annotations assigned to the `Austellungsdatum Label` and want to know which Spans would not be found when using the Whitespace Tokenizer. You can follow this example to find all the relevant Spans.

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer

my_project = Project(id_=TEST_PROJECT_ID)
category = my_project.categories[0]

tokenizer = WhitespaceTokenizer()

label = my_project.get_label_by_name('Austellungsdatum')

spans_not_found = label.spans_not_found_by_tokenizer(tokenizer, categories=[category])

for span in spans_not_found:
    print(f"{span}: {span.offset_string}")
```

### Conclusion
In this tutorial, we have walked through the essentials of tokenization. We have shown how the Konfuzio SDK can be configured to use different strategies to chunk your input text into tokens, before it is further processed to extract data from it.

We have seen the `WhitespaceTokenizer`, splitting text into chunks delimeted by white spaces. We have seen the more complex `RogexTokenizer`, which can be configured to use regular expressions to define delimiters matching any arbitrary regular expression. Furthermore, we have shown how regular expressions can be automatically found to be then used by the `RegexTokenizer`. We have also seen the `Paragraph`- and `SentenceTokenizer`, delimiting text according paragraphs and sentences respectively, and their different modes of use: `line_distance` for simpler documents, and `detectron` for documents with a more complex structure.


### What's Next?

- ...
- ...

