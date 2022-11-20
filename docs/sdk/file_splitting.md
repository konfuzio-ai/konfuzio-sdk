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
        print('Page {} is first Page'.format(page.number))
    else:
        print('Page {} is non-first Page'.format(page.number))

for page in test_document.pages():
    print({span.offset_string for span in page.spans()}.intersection(true_first_page_spans))
    print(len({span.offset_string for span in page.spans()}.intersection(true_first_page_spans)))
```

### Intro

Not all PDFs that we process are always neatly scanned and organized. Sometimes we can have more than one actual Document in a single file; such files need to be properly processed and split into several independent Documents. 

In this post, we will look at a simple way to implement an algorithm for splitting a multi-file Document into the sub-Documents. Our approach is based on an assumption that we can go through all Pages of the Document and define splitting points. By splitting points, we mean Pages that are similar in contents to the first Pages in the Documents. 

### Quick explanation

First step for implemention is "training": we tokenize the Document (meaning that we split its text into parts, more specifically to this tutorial – into strings without line breaks) and gather Spans – the parts of the text in the Page (we'll cover these concepts later) – each of which is present on every first Page of each Documents in the training data. 

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
[Project](https://dev.konfuzio.com/sdk/sourcecode.html#project) is one of the SDK's concepts. It is essentially a dataset that contains Documents belonging to different Categories (we will cover these shortly afterwards) or not having any Category assigned. It can be initialized as shown in the block above. 

The Project can also be accessed via the Smartview, with URL typically looking like https://app.konfuzio.com/admin/server/document/?project=YOUR_PROJECT_ID_HERE.

If you have made some local changes to the Project and want to return to the initial version available at the server, or if you want to fetch the updates from the server, use the argument `update=True`.

Here are the some of properties and methods of the Project you might need when working with the SDK:
- `project.documents` – training Documents within the Project;
- `project.test_documents` – test Documents within the Project;
- `project._documents` – all the Documents available within the Project;
- `project.get_category_by_id(YOUR_CATEGORY_ID).documents()` – Documents filtered by a Category of your choice; 
- `project.get_document_by_id(YOUR_DOCUMENT_ID)` – access a particular Document from the Project if you know its ID.

------

After initializing the Project, we filter the Documents that will be used for the "training" described [above](#Intro).  
```
training_docs = project.get_category_by_id(771).documents()
```
This is done because this logic only functions within a single Category – Documents similar in contents will most likely have at least one common first-page Span.

To learn more about Documents and Categories, go through the following sections:

#### Document
[Document](https://dev.konfuzio.com/sdk/sourcecode.html#document) is one of the files that constitute a Project. It consists of Pages, each of which has a set of Spans and Annotations (these will be covered further, too), and can belong to a certain Category. 

A Document can be accessed by `project.get_document_by_id(YOUR_DOCUMENT_ID)` when its ID is known to you; otherwise, it is possible to iterate through the output of `project.documents` (or `test_documents`/`_documents`) to see which Documents are available and what IDs they have.

The Documents can also be accessed via the Smartview, with URL typically looking like https://app.konfuzio.com/projects/PROJECT_ID/docs/DOCUMENT_ID/bbox-annotations/.

Here are some of the properties and methods of the Document you might need when working with the SDK:
- `document.id_` – get an ID of the Document;
- `document.text` – get a full text of the Document;
- `document.pages()` – a list of Pages in the Document;
- `document.update()` – download a newer version of the Document from the Server in case you have made some changes in the Smartview;
- `document.get_images()` – download PNG images of the Pages in the Document; can be used if you wish to use the visual data for training your own models, for example;

------

#### Category
[Category](https://dev.konfuzio.com/sdk/sourcecode.html#category) is a group of Documents united by common feature or type, i.e. invoice or receipt.

To see all Categories in the Project, you can use `project.get_categories()`. 
To find a Category the Document belongs to, you can use `document.category`.

You can also observe all Categories available in the Project via the Smartview: they are listed on the Project's page in the menu on the right.

-------

Next step is creating lists for collecting first-page Spans and Spans from other Pages.  To learn more about concepts of the Span and the Page, read the sections under the code block.

We also need to initialize the tokenizer that we imported. The tokenizer goes through the Document's text and separates it into Spans. We will use it to process training Documents, as well as any input Document in testing. This is done to ensure that the Documents are processed in a similar manner and it will be possible to find common Spans. 

ConnectedTextTokenizer in particular separates the text by `\n`  whitespaces.
```
first_page_spans = []
not_first_page_spans = []

tokenizer = ConnectedTextTokenizer()
```

#### Page
[Page](https://dev.konfuzio.com/sdk/sourcecode.html#page) is a part of the Document. Here are some of the properties and methods of the Page you might need when working with the SDK:
- `page.text` – get text of the Page;
- `page.spans()` – get a list of Spans on the Page;
- `page.number` – get Page's number, starting from 1.

------

#### Span
[Span](https://dev.konfuzio.com/sdk/sourcecode.html#span) is a part of the Document's text without the line breaks. Each Span has `start_offset` and `end_offset` denoting its starting and finishing characters in `document.text`. 

To access Span's text, you can call `span.offset_string`. We are going to use it later when collecting the Spans from the Documents.

------

Next, we will iterate through Documents in the training dataset. Note that the more Documents we use, the less intersecting Spans we are likely to find, so if at the end of the tutorial you find that your first-Page Spans set is empty, try using a slice of the dataset instead of the whole like it is done in the example below. However, usually when ran on Documents within same Category, this algorithm should not return an empty set; if that is the case, you might want to check if your data is consistent (i.e. not in different languages, no occurences of other Categories).

For each Document, we create a `deepcopy` .  Deepcopied object is identical to the one being copied but it is not referencing the original one or pointing towards it, unlike when simple `copy`  or variable reassignment is used. Deepcopied Document does not have Annotations (read about them under the code block).

Then, we run tokenizer on the deepcopied Document, which creates a set of Spans for it, and iterate through the Document's Pages.

If the Page is first, we append a set of its Span offset strings to `first_page_spans`; if it is not first, a set of its Span offset strings is appended to `not_first_page_spans`. We use offset_springs and not Spans themselves because while a same string can occur on different Pages, it might not necessarily be in the same position (with same start_offset and end_offset) and thus would not be counted as similar when compared to the Spans of an input Document.
```
for doc in training_docs[:5]:
    document_without_human_annotations = deepcopy(doc)
    doc = tokenizer.tokenize(document_without_human_annotations)
    for page in doc.pages():
        if page.number == 1:
            first_page_spans.append({span.offset_string for span in page.spans()})
        else:
            not_first_page_spans.append({span.offset_string for span in page.spans()})
```

#### Annotation 
[Annotation](https://dev.konfuzio.com/sdk/sourcecode.html#annotation) is a combination of Spans that has a certain Label  (i.e. Issue_Date, Auszahlungsbetrag) assigned to it. They typically denote a certain type of entity that is found in the text. Annotations can be predicted by AI or human-added. 

Like Spans, Annotations also have `start_offset` and `end_offset` denoting the starting and the ending characters. To access the text under the Annotation, call `annotation.offset_string`.

To see the Annotation in the Smartview, you can call `annotation.get_link()` and open the returned URL. 

----

After gathering all Span sets into two lists, we need to search for the Spans unique to the first Pages only. For that, we append an empty set to both of the lists (in case any of them are empty) and then apply `set.intersection` to both them. 

We use an asterisk before the list because we pass the sets in it as `args` – in other words, as multiple arguments, because we need to search for intersections between all of the sets.

After finding the intersections, we deduct Spans of non-first Pages from Spans of first Pages, thus obtaining a set of Spans that only appear at first Pages. 
```
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
```
test_document = project.get_category_by_id(771).test_documents()[0]
test_document = tokenizer.tokenize(deepcopy(test_document))
for page in test_document.pages():
    if len({span.offset_string for span in page.spans()}.intersection(true_first_page_spans)) > 0:
        print('Page {} is first Page'.format(page.number))
    else:
        print('Page {} is non-first Page'.format(page.number))

# output:
Page 1 is first Page
```
After tokenizing it in a manner similar to that of the training data, we go through its Pages (currently it's a one-page Document) and look whether at least one Span from the Page is present in our resulting "gold" set of unique first-page Spans. As we see, the Page is correctly labeled first.

Let's see how many intersections have been found:
```
for page in test_document.pages():
    print(len({span.offset_string for span in page.spans()}.intersection(true_first_page_spans)))

# output:
27
```
We can see that there is definitely more than one intersection with the list of the unique Spans, which further proves that the prediction is correct.