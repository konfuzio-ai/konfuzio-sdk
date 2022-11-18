## Splitting for multi-file Documents: Step-by-step guide

```
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from copy import deepcopy

project = Project(id_=226)

training_docs = project.get_category_by_id(771).documents()

first_page_spans = []
not_first_page_spans = []

tokenizer = ConnectedTextTokenizer()

for doc in training_docs[:20]:
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
print(true_first_page_spans - true_not_first_page_spans)

# add actual example of using the logic
```

### Intro

Not all PDFs that we process are always neatly scanned and organized. Sometimes we can have more than one actual Document in a single file; such files need to be properly processed and split into several independent Documents. 

In this post, we will look at a simple way to implement an algorithm for splitting a multi-file Document into the sub-Documents. Our approach is based on an assumption that we can go through all Pages of the Document and define splitting points. By splitting points, we mean Pages that are similar in contents to the first Pages in the Documents. 

### Quick explanation

`revisit the explanation for tokenize`

First step for implemention is "training": we tokenize the Document (meaning that we split its text into the parts) gather Spans – the parts of the text in the Page (we'll cover these concepts later) – each of which is present on every first Page of each Documents in the training data. 

Then, for every input Document's Page, we determine if it's first or not by going through its Spans and comparing them to the set of Spans collected in the first step. If we have at least one Span of intersection between the current Page and the Spans from the first step, we believe it is the first Page.

### Step-by-step explanation

Let's start with making all the necessary imports and initializing our Project:
```
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from copy import deepcopy

project = Project(id_=226) # any ID available to you can be here
```

To learn about Project, go through this section:

#### Project
Project is one of the SDK's concepts. It is essentially a dataset that contains Documents belonging to different Categories (we will cover these shortly afterwards) or not having any Category assigned. It can be initialized as shown in the block above. 

The Project can also be accessed via the Smartview, with URL typically looking like https://app.konfuzio.com/admin/server/document/?project=YOUR_PROJECT_ID_HERE.

If you have made some local changes to the Project and want to return to the initial version available at the server, or if you want to fetch the updates from the server, use the argument `update=True`.

Here are the some of properties and methods of the Project you might need when working with the SDK:
- `project.documents` – training Documents within the Project;
- `project.test_documents` – test Documents within the Project;
- `project._documents` – all the Documents available within the Project;
- `project.get_category_by_id(YOUR_CATEGORY_ID).documents()` – Documents filtered by a Category of your choice; 
- `project.get_document_by_id(YOUR_DOCUMENT_ID)` – access a particular Document from the Project if you know its ID.

After initializing the Project, we filter the Documents that will be used for the "training" described [above](#Intro).  
```
training_docs = project.get_category_by_id(771).documents()
```
This is done because this logic only functions within a single Category – Documents similar in contents will most likely have at least one common first-page Span.

To learn more about Documents and Categories, go through the following sections:

#### Document
Document is one of the files that constitute a Project. It consists of Pages, each of which has a set of Spans and Annotations (these will be covered further, too), and can belong to a certain Category. 

A Document can be accessed by `project.get_document_by_id(YOUR_DOCUMENT_ID)` when its ID is known to you; otherwise, it is possible to iterate through the output of `project.documents` (or `test_documents`/`_documents`) to see which Documents are available and what IDs they have.

The Documents can also be accessed via the Smartview, with URL typically looking like https://app.konfuzio.com/projects/PROJECT_ID/docs/DOCUMENT_ID/bbox-annotations/.

Here are some of the properties and methods of the Document you might need when working with the SDK:
- `document.text` – get a full text of the Document;
- `document.pages()` – a list of Pages in the Document;
- `document.update()` – download a newer version of the Document from the Server in case you have made some changes in the Smartview;
- `document.get_images()` – download PNG images of the Pages in the Document; can be used if you wish to use the visual data for training your own models, for example;

#### Category


explanation about Spans

explanation about tokenizing in general and connectedtexttokenizer in particular
```
first_page_spans = []
not_first_page_spans = []

tokenizer = ConnectedTextTokenizer()
```

#### Span

description of Span concept 

explanation of why slicing might be necessary
what is deepcopy here and why do we need here (`Document.__deepcopy__` – why not use this? ask)
what is the output of tokenizer.tokenize
```
for doc in training_docs[:20]:
    document_without_human_annotations = deepcopy(doc)
    doc = tokenizer.tokenize(document_without_human_annotations)
    for page in doc.pages():
        if page.number == 1:
            first_page_spans.append({span.offset_string for span in page.spans()})
        else:
            not_first_page_spans.append({span.offset_string for span in page.spans()})
```
what do we append and under what condition, for which purpose and why sets and why not spans themselves but rather offset_strings

#### Page

description of Page concept

explanation of searching for intersections 
explanation of usage of asterisks 
```
if not first_page_spans:
    first_page_spans.append(set())
true_first_page_spans = set.intersection(*first_page_spans)
if not not_first_page_spans:
    not_first_page_spans.append(set())
true_not_first_page_spans = set.intersection(*not_first_page_spans)
print(true_first_page_spans - true_not_first_page_spans)
```
explanation of result of deduction and how we aim at using it 

add example usage after the existent code chunk, show how the intersection is found and what exactly was the intersection Span(s)