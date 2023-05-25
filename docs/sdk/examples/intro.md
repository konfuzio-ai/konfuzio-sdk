## Example Usage

Make sure to set up your Project (so that you can retrieve the Project ID) using our [Konfuzio Guide](https://help.konfuzio.com/tutorials/quickstart/index.html).

### Project

Retrieve all information available for your Project:

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 15
   :dedent: 4

The information will be stored in the folder that you defined to allocate the data in the package initialization.
A subfolder will be created for each Document in the Project.

Every time that there are changes in the Project in the Konfuzio Server, the local Project can be updated this way:

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 17
   :dedent: 4

To make sure that your Project is loaded with all the latest data:

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 18
   :dedent: 4

### Documents

To access the Documents in the Project you can use:

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 20
   :dedent: 4


By default, it will get the Documents with training status (dataset_status = 2). The code for the status is:

- None: 0
- Preparation: 1
- Training: 2
- Test: 3
- Excluded: 4

The Test Documents can be accessed directly by:

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 21
   :dedent: 4

For more details, you can check out the [Project documentation](https://dev.konfuzio.com/sdk/sourcecode.html#project).


By default, you get 4 files for each Document that contain information of the text, pages, annotation sets and 
annotations. You can see these files inside the Document folder.

**document.txt** - Contains the text of the Document. If OCR was used, it will correspond to the result from the OCR.

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

**annotation_sets.json5** - Contains information of each section in the Document (for example, their ids and Label Sets).

```
[
  {
    "id": 78730,
    "position": 1,
    "section_label": 63
  },
  {
    "id": 292092,
    "position": 1,
    "section_label": 64
  }
]
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
    "get_created_by": "user@konfuzio.com",
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

#### Upload Document

This section details how to upload a new file in your Project using the Konfuzio SDK. The files must be of types specified 
in the Supported File Types. Here, we're primarily focusing on the `Document.from_file` method to create a 
[Konfuzio Document](https://dev.konfuzio.com/sdk/sourcecode.html#document).

A Konfuzio Document is an object representing the file you upload, it will contain the OCR (Optical Character Recognition) 
information of the file once processed by Konfuzio's server.

##### Synchronous and Asynchronous Upload

You have two options for uploading your file: a synchronous method and an asynchronous method. The method is determined 
by the `sync` parameter in the `from_file` function.

1. **Synchronous upload (sync=True)**: The file is uploaded to the Konfuzio servers, and the function waits for the 
document to be processed. Once done, it returns a Document object with the OCR information. This is useful if you want 
to start working with the document immediately after the OCR processing is completed.

2. **Asynchronous upload (sync=False)**: The function immediately returns an empty Document object after initiating the 
upload. The OCR processing takes place in the background. This method is advantageous when uploading a large file or a 
large number of files, as it doesn't require waiting for each file's processing to complete.

In the asynchronous upload, you can check the status of the document processing using the `document.update()` method on 
the returned Document object. If the document is ready, this method will update the Document object with the OCR information.

It's important to note that if the document is not ready, you may need to call `document.update()` again at a later time. 
This could be done manually or by setting up a looping mechanism depending on your application's workflow.

Here's an example of how to use the `from_file` method with `sync` set to `True`:

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 48
   :dedent: 4

##### Timeout Parameter

When making a server request, there's a default timeout value of 2 minutes. This means that if the server doesn't respond 
within 2 minutes, the operation will stop waiting for a response and return an error. If you're uploading a larger file, 
it might take more time to process, and the default timeout value might not be sufficient. In such a case, you can 
increase the timeout by setting the timeout parameter to a higher value.

Here's an example on how to do that:

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 50
   :dedent: 4

Another option for larger files, or if you are trying to upload a large number of files and don't want to wait for 
them to be processed would be to use the asynchronous option which returns an empty Document object. You can then 
use the `update` method to check if the Document is ready and the OCR processing is done:

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 52
   :dedent: 4

You can update the Document and check if the OCR output is available like this:

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 53-56
   :dedent: 4

#### Modify Document

If you would like to use the SDK to modify some Document's meta-data like the dataset status or the assignee, you can do
it like this:

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 61-64
   :dedent: 4

#### Update Document
If there are changes in the Document in the Konfuzio Server, you can update your local version of the Document with:

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 53
   :dedent: 4

If a Document is part of the Training or Test set, you can also update it by updating the entire Project via
`project.get(update=True)`. However, for Projects with many Documents it can be faster to update only the relevant Documents.

#### Download PDFs
To get the PDFs of the Documents, you can use `get_file()`.

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 25-26
   :dedent: 4

This will download the OCR version of the Document which contains the text, the Bounding Boxes
information of the characters and the image of the Document.

In the Document folder, you will see a new file with the original name followed by "_ocr".

If you want to original version of the Document (without OCR) you can use `ocr_version=False`.

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 28-29
   :dedent: 4

In the Document folder, you will see a new file with the original name.

#### Download pages as images
To get the Pages of the Document as png images, you can use `get_images()`.

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 32-33
   :dedent: 4

You will get one png image named "page_number_of_page.png" for each Page in the Document.

#### Download bounding boxes of the characters
To get the Bounding Boxes information of the characters, you can use `get_bbox()`.

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 37-38
   :dedent: 4

You will get a file named "bbox.json5".

After downloading these files, the paths to them will also become available in the Project instance.
For example, you can get the path to the file with the Document text with:

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 42
   :dedent: 4

#### Delete Document

##### Delete Document Locally
To locally delete a Document, you can use:

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 45
   :dedent: 4

The Document will be deleted from your local data folder, but it will remain in the Konfuzio Server.
If you want to get it again you can update the Project.

##### Delete Document Online

If you would like to delete a Document in the remote server you can simply use the `Document.delete` method the `delete_online` setting set to `True`. You can only delete Documents with a dataset status of None (0). **Be careful!** Once the Document is deleted online, we will have no way of recovering it. 

.. literalinclude:: /sdk/boilerplates/test_get_started.py
   :language: python
   :lines: 59,64,63,67
   :dedent: 4

If `delete_online` is set to False (the default), the Document will only be deleted on your local machine, and will be 
reloaded next time you load the Project, or if you run the `Project.init_or_update_document` method directly.
