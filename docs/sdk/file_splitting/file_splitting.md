## Splitting for multi-file Documents: Step-by-step guide

### Intro

Not all multipage files that we process are always neatly scanned and organized. Sometimes we can have more than one actual 
Document in a stream pf pages; such files need to be properly processed and split into several independent Documents. 

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

Let's start with making all the necessary imports and initializing our Project:
```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from copy import deepcopy

project = Project(id_=YOUR_PROJECT_ID) # any ID available to you can be here
```

After initializing the Project, we filter the Documents that will be used for the "training" described [above](#Intro).  
```python
training_docs = project.get_category_by_id(YOUR_CATEGORY_ID).documents()
```
This is done because this logic only functions within a single Category – Documents similar in contents will most likely have at least one common first-page Span.

Next step is creating lists for collecting first-page Spans and Spans from other Pages.  

We also need to initialize the Tokenizer that we imported. The Tokenizer goes through the Document's text and separates it into Spans. We will use it to process training Documents, as well as any Document that will undergo splitting. This is done to ensure that texts in all of the Documents are split using the same logic (tokenization by separating on `\n` whitespaces by ConnectedTextTokenizer, in particular) and it will be possible to find common Spans. 

```python
first_page_spans = []
not_first_page_spans = []

tokenizer = ConnectedTextTokenizer()
```

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

Next, we will iterate through Documents in the training dataset. 

For each Document, we create a `deepcopy` .  Deepcopied object is identical to the one being copied but it is not referencing the original one or pointing towards it, unlike when simple `copy`  or variable reassignment is used. Deepcopied Document does not have Annotations.

Then, we run Tokenizer on the deepcopied Document, which creates a set of Spans for it, and iterate through the Document's Pages.

If the Page is first, we append a set of its Span offset strings to `first_page_spans`; if it is not first, a set of its Span offset strings is appended to `not_first_page_spans`. We use offset_springs and not Spans themselves because while a same string can occur on different Pages, it might not necessarily be in the same position (with same start_offset and end_offset) and thus would not be counted as similar when compared to the Spans of an input Document.
```python
for doc in training_docs[:5]:
    document_without_human_annotations = deepcopy(doc)
    doc = tokenizer.tokenize(document_without_human_annotations)
    for page in doc.pages():
        if page.number == 1:
            first_page_spans.append({span.offset_string for span in page.spans()})
        else:
            not_first_page_spans.append({span.offset_string for span in page.spans()})
```


After gathering all Span sets into two lists, we need to search for the Spans unique to the first Pages only. For that, we append an empty set to both of the lists (in case any of them are empty) and then apply `set.intersection` to both them. 

We use an asterisk before the list because we pass the sets in it as `args` – in other words, as multiple arguments, because we need to search for intersections between all of the sets.

After finding the intersections, we deduct Spans of non-first Pages from Spans of first Pages, thus obtaining a set of Spans that only appear at first Pages. 
```python
if not first_page_spans:
    first_page_spans.append(set())
true_first_page_spans = set.intersection(*first_page_spans)
if not not_first_page_spans:
    not_first_page_spans.append(set())
true_not_first_page_spans = set.intersection(*not_first_page_spans)
true_first_page_spans = true_first_page_spans - true_not_first_page_spans
```
So, it is safe to assume that if a Page has at least one Span from the resulting set, it is a first 
Page.

Let's test our algorithm on a test Document from the same Category:
```python
test_document = project.get_category_by_id(YOUR_CATEGORY_ID).test_documents()[0]
test_document = tokenizer.tokenize(deepcopy(test_document))
for page in test_document.pages():
    if len({span.offset_string for span in page.spans()}.intersection(true_first_page_spans)) > 0:
        print('Page {} is predicted as a first Page'.format(page.number))
    else:
        print('Page {} is predicted as a non-first Page'.format(page.number))
```
After tokenizing it in a manner similar to that of the training data, we go through its Pages (currently it's a one-page Document) and look whether at least one Span from the Page is present in our resulting set of unique first-page Spans. As we see, the Page is correctly labeled first.

Let's see how many intersections have been found:
```python
for page in test_document.pages():
    print(len({span.offset_string for span in page.spans()}.intersection(true_first_page_spans)))
```

Full code:
```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from copy import deepcopy

project = Project(id_=YOUR_PROJECT_ID)

training_docs = project.get_category_by_id(YOUR_CATEGORY_ID).documents()

first_page_spans = []
not_first_page_spans = []

tokenizer = ConnectedTextTokenizer()

for doc in training_docs[:5]:
    document_without_human_annotations = deepcopy(doc)
    doc = tokenizer.tokenize(document_without_human_annotations)
    for page in doc.pages():
        if page.number == 1:
            first_page_spans.append({span.offset_string for span in page.spans()})
        else:
            not_first_page_spans.append({span.offset_string for span in page.spans()})

if not first_page_spans:
    first_page_spans.append(set())
true_first_page_spans = set.intersection(*first_page_spans)
if not not_first_page_spans:
    not_first_page_spans.append(set())
true_not_first_page_spans = set.intersection(*not_first_page_spans)
true_first_page_spans = true_first_page_spans - true_not_first_page_spans

test_document = project.get_category_by_id(771).test_documents()[0]
test_document = tokenizer.tokenize(deepcopy(test_document))
for page in test_document.pages():
    if len({span.offset_string for span in page.spans()}.intersection(true_first_page_spans)) > 0:
        print('Page {} is predicted as a first Page'.format(page.number))
    else:
        print('Page {} is predicted as a non-first Page'.format(page.number))

for page in test_document.pages():
    print({span.offset_string for span in page.spans()}.intersection(true_first_page_spans))
    print(len({span.offset_string for span in page.spans()}.intersection(true_first_page_spans)))
```