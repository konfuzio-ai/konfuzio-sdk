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
| :ledger: [Docs](https://dev.konfuzio.com)                                                     | Read the docs                                                 |
| :floppy_disk: [Installation](https://github.com/konfuzio-ai/konfuzio-sdk#installation)        | How to install the Konfuzio SDK                               |
| :mortar_board: [Tutorials](https://dev.konfuzio.com/sdk/tutorials.html)             | See what the Konfuzio SDK can do with our Notebooks & Scripts |
| :beginner: [Explanations](https://dev.konfuzio.com/sdk/explanations.html)                     | Here are links to teaching material about the Konfuzio SDK.   |
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

Find the full installation guide [here](https://dev.konfuzio.com/sdk/examples/index.html#install-sdk)
or setup PyCharm as described [here](https://dev.konfuzio.com/sdk/quickstart_pycharm.html).

## CLI

We provide the basic function to create a new Project via CLI:

`konfuzio_sdk create_project YOUR_PROJECT_NAME`

You will see "Project `{YOUR_PROJECT_NAME}` (ID `{YOUR_PROJECT_ID}`) was created successfully!" printed.

And download any project via the id:

`konfuzio_sdk export_project YOUR_PROJECT_ID`

## Tutorials

### Basics

```python
from konfuzio_sdk.data import Project, Document

# Initialize the project:
my_project = Project(id_='YOUR_PROJECT_ID')

# Get any project online
doc: Document = my_project.get_document_by_id('DOCUMENT_ID_ONLNIE')

# Get the Annotations in a Document
doc.annotations()

# Filter Annotations by Label
label = my_project.get_label_by_name('MY_OWN_LABEL_NAME')
doc.annotations(label=label)

# Or get all Annotations that belong to one Label
cat = p.get_category_by_id(4433)
label.annotations(categories=[cat])

# Force a project update. To save time documents will only be updated if they have changed.
my_project.update()
```

Find more examples in the [Tutorials](https://dev.konfuzio.com/sdk/tutorials.html).
