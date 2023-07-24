# Konfuzio SDK

![Downloads](https://pepy.tech/badge/konfuzio-sdk)

The Konfuzio Software Development Kit (Konfuzio SDK) provides a
[Python API](https://dev.konfuzio.com/sdk/sourcecode.html) to interact with the
[Konfuzio Server](https://dev.konfuzio.com/index.html#konfuzio-server).

## Features

The SDK allows you to retrieve visual and text features to build your own document models. Konfuzio Server serves as an
UI to define the data structure, manage training/test data and to deploy your models as API.

Function               | Public Host Free*                         | On-Site (Paid)      |
:--------------------- | :---------------------------------------- | :-------------------|
OCR Text               | :heavy_check_mark:                        |  :heavy_check_mark: |
OCR Handwriting        | :heavy_check_mark:                        |  :heavy_check_mark: |
Text Annotation        | :heavy_check_mark:                        |  :heavy_check_mark: |
PDF Annotation         | :heavy_check_mark:                        |  :heavy_check_mark: |
Image Annotation       | :heavy_check_mark:                        |  :heavy_check_mark: |
Table Annotation       | :heavy_check_mark:                        |  :heavy_check_mark: |
Download HOCR          | :heavy_check_mark:                        |  :heavy_check_mark: |
Download Images        | :heavy_check_mark:                        |  :heavy_check_mark: |
Download PDF with OCR  | :heavy_check_mark:                        |  :heavy_check_mark: |
Deploy AI models       | :heavy_multiplication_x:                  |  :heavy_check_mark: |

`*` Under fair use policy: We will impose 10 pages/hour throttling eventually.


|                                                                                               |                                                               |
|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| :ledger: [Docs](https://dev.konfuzio.com/sdk/index.html)                                                    | Read the docs                                                 |
| :floppy_disk: [Installation](https://github.com/konfuzio-ai/konfuzio-sdk#installation)        | How to install the Konfuzio SDK                               |
| :mortar_board: [Tutorials](https://dev.konfuzio.com/sdk/examples/examples.html)               | See what the Konfuzio SDK can do with our Notebooks & Scripts |
| :bulb: [Explanations](https://dev.konfuzio.com/sdk/explanations.html)                         | Here are links to teaching material about the Konfuzio SDK.   |
| :gear: [API Reference](https://dev.konfuzio.com/sdk/sourcecode.html)                          | Python classes, methods, and functions                        |
| :heart: [Contributing](https://dev.konfuzio.com/sdk/contribution.html)                        | Learn how to contribute!                                      |
| :bug: [Issue Tracker](https://github.com/konfuzio-ai/konfuzio-sdk/issues)                     | Report and monitor Konfuzio SDK issues                        |
| :telescope: [Changelog](https://github.com/konfuzio-ai/konfuzio-sdk/releases)                 | Review the release notes                                      |
| :newspaper: [MIT License](https://github.com/konfuzio-ai/konfuzio-sdk/blob/master/LICENSE.md) | Review the license                                            |

## Installation

As developer register on our [public HOST for free: https://app.konfuzio.com](https://app.konfuzio.com/accounts/signup/)

Then you can use pip to install Konfuzio SDK and run init:

    pip install konfuzio_sdk

    konfuzio_sdk init

The init will create a Token to connect to the Konfuzio Server. This will create variables `KONFUZIO_USER`,
`KONFUZIO_TOKEN` and `KONFUZIO_HOST` in an `.env` file in your working directory.

By default, the SDK is installed without the AI-related dependencies like `torch` or `transformers` and allows for using 
only the Data-related SDK concepts but not the AI models. To install the SDK with the AI components,
run the following command:
  ```
  pip install konfuzio_sdk[ai]
  ```

Find the full installation guide [here](https://dev.konfuzio.com/sdk/get_started.html#install-sdk)
or setup PyCharm as described [here](https://dev.konfuzio.com/sdk/quickstart_pycharm.html).

## CLI

We provide the basic function to create a new Project via CLI:

`konfuzio_sdk create_project YOUR_PROJECT_NAME`

You will see "Project `{YOUR_PROJECT_NAME}` (ID `{YOUR_PROJECT_ID}`) was created successfully!" printed.

And download any project via the id:

`konfuzio_sdk export_project YOUR_PROJECT_ID`

## Tutorials

You can find detailed examples about how to set up and run document AI pipelines in our 
[Tutorials](https://dev.konfuzio.com/sdk/tutorials.html), including:
- [Split a file into separate Documents](https://dev.konfuzio.com/sdk/tutorials.html#split-a-file-into-separate-documents)
- [Document Categorization](https://dev.konfuzio.com/sdk/tutorials.html#document-categorization)
- [Train a Konfuzio SDK Model to Extract Information From Payslip Documents](https://dev.konfuzio.com/sdk/tutorials.html#train-a-konfuzio-sdk-model-to-extract-information-from-payslip-documents)

### Basics

Here we show how to use the Konfuzio SDK to retrieve data hosted on a Konfuzio Server instance.

```python
from konfuzio_sdk.data import Project, Document

# Initialize the Project
YOUR_PROJECT_ID: int
my_project = Project(id_=YOUR_PROJECT_ID)

# Get any Document online
DOCUMENT_ID_ONLINE: int
doc: Document = my_project.get_document_by_id(DOCUMENT_ID_ONLINE)

# Get the Annotations in a Document
doc.annotations()

# Filter Annotations by Label
MY_OWN_LABEL_NAME: str
label = my_project.get_label_by_name(MY_OWN_LABEL_NAME)
doc.annotations(label=label)

# Or get all Annotations that belong to one Category
YOUR_CATEGORY_ID: int
category = my_project.get_category_by_id(YOUR_CATEGORY_ID)
label.annotations(categories=[category])

# Force a Project update. To save time Documents will only be updated if they have changed.
my_project.get(update=True)
```
