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

### Environment
You need to install the Konfuzio SDK before diving into the tutorial. \
To get up and running quickly, you can use our Colab Quick Start notebook. \
<a href="https://colab.research.google.com/github/konfuzio-ai/konfuzio-sdk/blob/master/notebooks/Quick_start_template_for_Konfuzio_SDK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

As an alternative you can follow the [installation section](../get_started.html#install-sdk) to install and initialize the Konfuzio SDK locally or on an environment of your choice.

## Example Usage

Make sure to set up your Project (so that you can retrieve the Project ID) using our [Konfuzio Guide](https://help.konfuzio.com/tutorials/quickstart/index.html).

### Project

Retrieve all information available for your Project:

```python tags=['remove-cell']
import logging

logging.getLogger("konfuzio_sdk").setLevel(logging.ERROR)
YOUR_PROJECT_ID = 46
```
```python 
from konfuzio_sdk.data import Project, Document

my_project = Project(id_=YOUR_PROJECT_ID)
```

The information will be stored in the folder that you defined to allocate the data in the package initialization.
A subfolder will be created for each Document in the Project.

Every time that there are changes in the Project in the Konfuzio Server, the local Project can be updated this way:

```python
my_project.get(update=True)
```

To make sure that your Project is loaded with all the latest data:

```python
my_project = Project(id_=YOUR_PROJECT_ID, update=True)
```

### Documents

Every Document has a status indicating in what stage of processing it is. The code for the Document status is:

   - Queuing for OCR: 0
   - Queuing for extraction: 1
   - Done: 2
   - Could not be processed: 111
   - OCR in progress: 10
   - Extraction in progress: 20
   - Queuing for categorization: 3
   - Categorization in progress: 30
   - Queuing for splitting: 4
   - Splitting in progress: 40
   - Waiting for splitting confirmation: 41

To access the Documents in the Project you can use:

```python
documents = my_project.documents
```

By default, it will get the Documents with training status (dataset_status = 2). The code for the dataset status is:

- None: 0
- Preparation: 1
- Training: 2
- Test: 3
- Excluded: 4

The Test Documents can be accessed directly by:

```python
test_documents = my_project.test_documents
```

For more details, you can check out the [Project Documentation](https://dev.konfuzio.com/sdk/sourcecode.html#project).


By default, you get 4 files for each Document that contain information of the text, Pages, Annotation Sets and 
Annotations. You can see these files inside the Document folder.

`.txt` file contains the text of the Document. If OCR was used, it will correspond to the result from the OCR.

```
                                                            x02   328927/10103/00104
Abrechnung  der Brutto/Netto-Bezüge   für Dezember 2018                   22.05.2018 Bat:  1

Personal-Nr.  Geburtsdatum ski Faktor  Ki,Frbtr.Konfession  ‚Freibetragjährl.! |Freibetrag mt! |DBA  iGleitzone  'St.-Tg.  VJuUr. üb. |Url. Anspr. Url.Tg.gen.  |Resturlaub
00104 150356 1  |     ‚ev                              30     400  3000       3400

SV-Nummer       |Krankenkasse                       KK%®|PGRS Bars  jum.SV-Tg. Anw. Tage |UrlaubTage Krankh. Tg. Fehlz. Tage

50150356B581 AOK  Bayern Die Gesundheitskas 157 101 1111 1 30

                                             Eintritt   ‚Austritt     Anw.Std.  Urlaub Std.  |Krankh. Std. |Fehlz. Std.

                                             170299  L L       l     L     l     l
 -                                       +  Steuer-ID       IMrB?       Zeitlohn Sta.|Überstd.  |Bez. Sta.
  Teststraße123
   12345 Testort                                   12345678911       \     ı     ı     \
                               B/N
               Pers.-Nr.  00104        x02
               Abt.-Nr. A12         10103          HinweisezurAbrechnung
```

**pages.json5** - Contains information of each Page of the Document (for example, their ids and sizes).

```
[
  {
    "id": 1923,
    "image": "/page/show/1923/",
    "number": 1,
    "original_size": [
      595.2,
      841.68
    ],
    "size": [
      1414,
      2000
    ]
  }
]
```

**annotation_sets.json5** - Contains information of each Annotation Set in the Document and Annotations that constitute
it.

```
{
    "id": 78730,
    "label_set": {
      "api_name": "Lohnabrechnung",
      "description": "",
      "has_multiple_annotation_sets": false,
      "id": 63,
      "name": "Lohnabrechnung"
    },
    "labels": [
      {
        "annotations": [
          {
            "confidence": 0.93,
            "created_by": "user@mail.com",
            "custom_offset_string": false,
            "Document": 44823,
            "id": 4420351,
            "is_correct": true,
            "normalized": 2189.07,
            "offset_string": "2.189,07",
            "offset_string_original": "2.189,07",
            "origin": "api.v2",
            "revised": false,
            "revised_by": null,
            "selection_bbox": {
              "page_index": 0,
              "x0": 516.48,
              "x1": 562.8,
              "y0": 76.829,
              "y1": 87.829
            },
            "span": [
              {
                "end_offset": 3785,
                "offset_string": "2.189,07",
                "offset_string_original": "2.189,07",
                "page_index": 0,
                "start_offset": 3777,
                "x0": 516.48,
                "x1": 562.8,
                "y0": 76.829,
                "y1": 87.829
              }
            ],
            "translated_string": null
          }
```

**annotations.json5** - Contains information of each Annotation in the Document (for example, their Labels and Bounding 
Boxes).

```
[
  {
    "accuracy": null,
    "bbox": {
      "bottom": 44.369,
      "line_index": 1,
      "page_index": 0,
      "top": 35.369,
      "x0": 468.48,
      "x1": 527.04,
      "y0": 797.311,
      "y1": 806.311
    },
    "bboxes": [
      {
        "bottom": 44.369,
        "end_offset": 169,
        "line_number": 2,
        "offset_string": "22.05.2018",
        "offset_string_original": "22.05.2018",
        "page_index": 0,
        "start_offset": 159,
        "top": 35.369,
        "x0": 468.48,
        "x1": 527.04,
        "y0": 797.311,
        "y1": 806.311
      }
    ],
    "created_by": 59,
    "custom_offset_string": false,
    "end_offset": 169,
    "get_created_by": "user@mail.com",
    "get_revised_by": "n/a",
    "id": 4419937,
    "is_correct": true,
    "label": 867,
    "label_data_type": "Date",
    "label_text": "Austellungsdatum",
    "label_threshold": 0.1,--
    "normalized": "2018-05-22",
    "offset_string": "22.05.2018",
    "offset_string_original": "22.05.2018",
    "revised": false,
    "revised_by": null,
    "section": 78730,
    "section_label_id": 63,
    "section_label_text": "Lohnabrechnung",
    "selection_bbox": {
      "bottom": 44.369,
      "line_index": 1,
      "page_index": 0,
      "top": 35.369,
      "x0": 468.48,
      "x1": 527.04,
      "y0": 797.311,
      "y1": 806.311
    },
    "start_offset": 159,
    "translated_string": null
  },
...
]
```

When needed, upon calling `Document.get_bbox()`, an additional file will be downloaded to the Document folder containing the Bounding Boxes information of the characters of the Document: **bbox.zip**. This file can be quite large, and therefore it will be compressed in the Zip format. The decompressed file is a JSON file where the keys correspond to the indices of the characters in the Document text. The value associated with each key contains the Bounding Box information of the character. For example, for character 1000 and 1002 we would have:
   
```
{
  "1000": {
    "adv": 2.58,
    "bottom": 128.13,
    "doctop": 118.13,
    "fontname": "GlyphLessFont",
    "height": 10.0,
    "line_number": 14,
    "object_type": "char",
    "page_number": 1,
    "size": 10.0,
    "text": "n",
    "top": 118.13,
    "upright": 1,
    "width": 2.58,
    "x0": 481.74,
    "x1": 484.32,
    "y0": 713.55,
    "y1": 723.55
  },
  "1002": {
    "adv": 2.64,
    "bottom": 128.13,
    "doctop": 118.13,
    "fontname": "GlyphLessFont",
    "height": 10.0,
    "line_number": 14,
    "object_type": "char",
    "page_number": 1,
    "size": 10.0,
    "text": "S",
    "top": 118.13,
    "upright": 1,
    "width": 2.64,
    "x0": 486.72,
    "x1": 489.36,
    "y0": 713.55,
    "y1": 723.55
  },
// ...
}
```

After downloading these files, their paths will become available in the Project instance.

You can get the path to the folder containing the Documents' folders with:

```python
my_project.documents_folder
```

And you can get the path to the file with the Document text with:
```python tags=['remove-cell']
document = my_project.get_document_by_id(44823)
```
```python
document.txt_file_path
```

#### Upload Document

Before you can upload a new file to your Project using the Konfuzio SDK, you must have completed the following steps:

1. Register for a Konfuzio account
2. Create a Project on Konfuzio
3. Install the Konfuzio SDK

For detailed instructions on these preliminary steps, refer above to the [Get Started guide](https://dev.konfuzio.com/sdk/get_started.html#get-started).

After completing the above steps, you can proceed with uploading a new file to your Project using the Konfuzio SDK. The 
files must be of types specified in the [Supported File Types](https://help.konfuzio.com/specification/supported_file_types/index.html). 
Here, we're focusing on the `Document.from_file` method to create a [Konfuzio Document](https://dev.konfuzio.com/sdk/sourcecode.html#document).

A Konfuzio Document is an object representing the file you upload, it will contain the OCR (Optical Character Recognition) 
information of the file once processed by Konfuzio's server.

###### Synchronous and Asynchronous Upload

You have two options for uploading your file: a synchronous method and an asynchronous method. The method is determined 
by the `sync` parameter in the `from_file` method.

1. **Synchronous upload (sync=True)**: The file is uploaded to the Konfuzio servers, and the method waits for the 
file to be processed. Once done, it returns a Document object with the OCR information. This is useful if you want 
to start working with the Document immediately after the OCR processing is completed.

   Here's an example of how to use the `from_file` method with `sync` set to `True`:
```python tags=['remove-cell']
import time
FILE_PATH = 'tests/test_data/pdf.pdf'
ASSIGNEE_ID = None
```
```python tags=["skip-execution", "nbval-skip"]
document = Document.from_file(FILE_PATH, project=my_project, sync=True)
```
```python tags=['remove-cell']
document = my_project._documents[-1]
document.dataset_status = 0
document.delete(delete_online=True)
```

2. **Asynchronous upload (sync=False)**: With this setting, the method immediately returns an empty Document object 
after initiating the upload. The OCR processing takes place in the background. This method is advantageous when 
uploading a large file or a large number of files, as it doesn't require waiting for each file's processing to complete.

   Here is how to use the asynchronous method:

```python tags=["skip-execution", "nbval-skip"]
document = Document.from_file(FILE_PATH, project=my_project, sync=False)
```

After asynchronous upload, you can check the status of the Document processing using the `Document.update()` method on 
the returned Document object. If the Document is ready, this method will update the Document object with the OCR information.

It's important to note that if the Document is not ready, you may need to call `Document.update()` again at a later time. 
This could be done manually or by setting up a looping mechanism depending on your application's workflow.

To check if the Document is ready and update it with the OCR information, you can implement a custom pulling strategy 
like this:

```python tags=["skip-execution", "nbval-skip"]
for i in range(2):
    document.update()
    if document.ocr_ready is True:
        print(document.text)
        break
    time.sleep(i * 10 + 3)
```

For a more sophisticated pulling method for asynchronously uploaded Documents using the callback function, you can 
checkout our :ref:`tutorial on how to use ngrok to receive callbacks from the Konfuzio Server<async_upload_with_callback>`.

###### Timeout Parameter

When making a server request, there's a default timeout value of 2 minutes. This means that if the server doesn't respond 
within 2 minutes, the operation will stop waiting for a response and return an error. If you're uploading a larger file, 
it might take more time to process, and the default timeout value might not be sufficient. In such a case, you can 
increase the timeout by setting the timeout parameter to a higher value.

```python tags=["skip-execution", "nbval-skip"]
document = Document.from_file(FILE_PATH, project=my_project, timeout=300, sync=True)
```


#### Modify Document

If you would like to use the SDK to modify some Document's meta-data like the dataset status or the assignee, you can do
it like this:

```python tags=["skip-execution", "nbval-skip"]
document.assignee = ASSIGNEE_ID
document.dataset_status = 2

document.save_meta_data()
```

#### Update Document
If there are changes in the Document in the Konfuzio Server, you can update your local version of the Document with:

```python tags=["skip-execution", "nbval-skip"]
document.update()
```

If a Document is part of the Training or Test set, you can also update it by updating the entire Project via
`Project.get(update=True)`. However, for Projects with many Documents it can be faster to update only the relevant Documents.

#### Download PDFs
To get the PDFs of the Documents, you can use `get_file()`.

```python tags=["skip-execution", "nbval-skip"]
for document in my_project.documents:
    document.get_file()
```

This will download the OCR version of the Document which contains the text, the Bounding Boxes
information of the characters and the image of the Document.

In the Document folder, you will see a new file with the original name followed by "_ocr".

If you want to original version of the Document (without OCR) you can use `ocr_version=False`.

```python tags=["skip-execution", "nbval-skip"]
for document in my_project.documents:
    document.get_file(ocr_version=False)
```

In the Document folder, you will see a new file with the original name.

#### Download pages as images
To get the Pages of the Document as png images, you can use `get_images()`.

```python tags=["skip-execution", "nbval-skip"]
for document in my_project.documents:
    document.get_images()
```

You will get one png image named "page_number_of_page.png" for each Page in the Document.

#### Download bounding boxes of the characters
To get the Bounding Boxes information of the characters, you can use `get_bbox()`.

```python tags=["skip-execution", "nbval-skip"]
for document in my_project.documents:
    document.get_bbox()
```

You will get a file named "bbox.zip" in the Document folder. This file contains the "bbox.json5" file. You can find the
path to the zip file in the Document instance with:

```python tags=["skip-execution", "nbval-skip"]
document.bbox_file_path
```

#### Delete Document

##### Delete Document Locally
To locally delete a Document, you can use:

```python tags=["skip-execution", "nbval-skip"]
document.delete()
```

The Document will be deleted from your local data folder, but it will remain in the Konfuzio Server.
If you want to get it again you can update the Project.

##### Delete Document Online

If you would like to delete a Document in the remote server you can simply use the `Document.delete` method the `delete_online` setting set to `True`. You can only delete Documents with a dataset status of None (0). **Be careful!** Once the Document is deleted online, we will have no way of recovering it. 

```python tags=["skip-execution", "nbval-skip"]
document.delete(delete_online=True)
```

If `delete_online` is set to False (the default), the Document will only be deleted on your local machine, and will be 
reloaded next time you load the Project.