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

## Prepare Training and Testing Data

---

**Prerequisites:**
- Basic knowledge about what training and testing means in the context of AI models
- Be familiar with uploading [Documents](https://help.konfuzio.com/modules/documents/index.html) using the Konfuzio UI
- Have access to a [Konfuzio Project](https://help.konfuzio.com/modules/projects/index.html) or have the permissions to create one

**Difficulty:** Easy

**Goal:** Be able to programmatically upload Documents within a Konfuzio Project.

---

### Environment
You need to install the Konfuzio SDK before diving into the tutorial. \
To get up and running quickly, you can use our Colab Quick Start notebook. \
<a href="https://colab.research.google.com/github/konfuzio-ai/konfuzio-sdk/blob/master/notebooks/Quick_start_template_for_Konfuzio_SDK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

As an alternative you can follow the [installation section](../get_started.html#install-sdk) to install and initialize the Konfuzio SDK locally or on an environment of your choice.

### Introduction
Before training an AI model, Documents for training and testing need to be uploaded within a Konfuzio Project. This can be done in two ways: using the server interface, detailed instruction to upload Documents are [here](https://help.konfuzio.com/modules/documents/index.html). Alternatively, to reduce the manual work of uploading many Documents at once the instructions described in this tutorial can be followed.


### Upload Documents


We start by defining a path where a PDF Document exists.

```python tags=["skip-execution", "nbval-skip"]
FILE_PATH_1 = 'path/to/pdf_file1.pdf'
FILE_PATH_2 = 'path/to/pdf_file2.pdf'
FILE_PATH_3 = 'path/to/pdf_file3.pdf'
```

```python tags=["remove-cell"]
# This cell gets removed when the notebook is compiled as markdown
FILE_PATH = '../../../../tests/test_data/pdf.pdf'

# Use the same file for the sake of local testing
FILE_PATH_1 = FILE_PATH_2 = FILE_PATH_3 = FILE_PATH
```

We import the id of the default test Project, as well as the libraries we need:

```python tags=["remove-cell"]
# This is necessary to make sure we can import from 'tests'
import sys
from konfuzio_sdk.api import get_project_list
sys.path.insert(0, '../../../../')

from tests.variables import TEST_SNAPSHOT_ID

projects = get_project_list()
# we want to get the last instance of a project restored from a snapshot because creating a new one each time takes longer 
TEST_PROJECT_ID = next(project['id'] for project in reversed(projects) if TEST_SNAPSHOT_ID in project['name'])
```

```python
from konfuzio_sdk.data import Project, Document
```

We now initialize a Konfuzio Project object and a list of paths with the PDFs of interest: 

```python tags=["remove-output"]
project = Project(id_=TEST_PROJECT_ID)
file_paths = [FILE_PATH_1, FILE_PATH_2, FILE_PATH_3]
```

Then, create a new Document from each PDF in the list of paths `file_paths`:

```python
for document_path in file_paths:
    _ = Document.from_file(document_path, project=project, sync=True)
    print(f'Document {_.id_} successfully created.')
```

The `Document.from_file` method uploads a new Document to the Konfuzio server. The [documentation](https://dev.konfuzio.com/sdk/sourcecode.html#document) provides an overview of the returned values and optional paramenters for this method.

<!-- #region link="get_started.html#modify-document" -->
### Optional: Alter Document Status
A Document in the Konfuzio Server can be assigned a Status defining if the Document belongs to the Training or Testing data-set. Setting a Status for a Document is necessary to determine which Documents will be used for Training (status 1) and which will be used for Testing (status 2). More information can be found in the [Konfuzio Manual](https://help.konfuzio.com/modules/documents/index.html#id1), and an example can be found [here](https://dev.konfuzio.com/sdk/get_started.html#modify-document).
<!-- #endregion -->

### Conclusion
In this tutorial, we have walked through the essential steps for programmatically uploading PDFs to the Konfuzio Server. Below is the full code to accomplish this task:

```python tags=["skip-execution", "nbval-skip"]
from konfuzio_sdk.data import Project, Document

FILE_PATH_1 = 'path/to/pdf_file1.pdf'
FILE_PATH_2 = 'path/to/pdf_file2.pdf'
FILE_PATH_3 = 'path/to/pdf_file3.pdf'

project = Project(id_=TEST_PROJECT_ID)
file_paths = [FILE_PATH_1, FILE_PATH_2, FILE_PATH_3]

for document_path in file_paths:
    _ = Document.from_file(document_path, project=project, sync=False)
    print(f'Document {_.id_} successfully created.')
```

