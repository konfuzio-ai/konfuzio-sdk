## Example Usage

### Project

Retrieve all information available for your project:

```python
my_project = Project(id_=YOUR_PROJECT_ID)
```

The information will be stored in the folder that you defined to allocate the data in the package initialization.
A subfolder will be created for each document in the project.

Every time that there are changes in the project in the Konfuzio Server, the local project can be updated with:

```python
my_project.update()
```

### Documents

To access the documents in the project you can use:

```python
documents = my_project.get_documents_by_status()
```

By default, it will get the documents without dataset status (dataset_status = 0 (None)).
You can specify another dataset status with the argument 'dataset_statuses'. The code for the status is:

- None: 0
- Preparation: 1
- Training: 2
- Test: 3
- Excluded: 4

For example, to get all documents in the project, you can do:

```python
documents = my_project.get_documents_by_status(dataset_statuses=[0, 1, 2, 3, 4])
```

The training and test documents can be accessed directly by:

```python
training_documents = my_project.documents
test_documents = my_project.test_documents
```

By default, you get 4 files for each document that contain information of the text, pages, annotation sets and annotations.
You can see these files inside the document folder.

**document.txt** - Contains the text of the document. If OCR was used, it will correspond to the result from the OCR.

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

**pages.json5** - Contains information of each page of the document (for example, their ids and sizes).

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

**annotation_sets.json5** - Contains information of each section in the document (for example, their ids and label sets).

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

**annotations.json5** - Contains information of each annotation in the document (for example, their labels and bounding boxes).

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

#### Download PDFs
To get the PDFs of the documents, you can use **get_file()**.

```python
for document in my_project.documents:
    document.get_file()
```

This will download the OCR version of the document which contains the text, the bounding boxes
information of the characters and the image of the document.

In the document folder, you will see a new file with the original name followed by "_ocr".

If you want to original version of the document (without OCR) you can use **ocr_version=False**.

```python
for document in my_project.documents:
    document.get_file(ocr_version=False)
```

In the document folder, you will see a new file with the original name.

#### Download pages as images
To get the pages of the document as png images, you can use **get_images()**.

```python
for document in my_project.documents:
    document.get_images()
```

You will get one png image named "page_number_of_page.png" for each page in the document.

#### Download bounding boxes of the characters
To get the bounding boxes information of the characters, you can use **get_bbox()**.

```python
for document in my_project.documents:
    document.get_bbox()
```

You will get a file named "bbox.json5".

After downloading these files, the paths to them will also become available in the project instance.
For example, you can get the path to the file with the document text with:

```python
my_project.txt_file_path
```

#### Update Document
If there are changes in the document in the Konfuzio Server, you can update the document with:

```python
document.update()
```

If a document is part of the training or test set, you can also update it by updating the entire project via
project.update(). However, for projects with many documents it can be faster to update only the relevant documents.

#### Upload Document
You can upload a document via SDK. Create a Document instance and save it.
The document will be uploaded to the Konfuzio Server.

```python
document = Document(file_path=<path_to_the_file>, project=my_project)
document.save()
```

By default, the document is uploaded with the dataset status "None". If there is only one category in the project, the
document will assume that category. If there is more than one category in the project, the document is uploaded without
a category.

You can specify both these parameters when you upload the document by passing the correspondent code for the dataset
status (see code [here](https://app.konfuzio.com/v2/swagger/#/docs/docs_create)) and the ID of the category.

```python
document = Document(file_path=<path_to_the_file>, project=my_project,
                    dataset_status=<dataset_status_code>, category_template=<category_id>)
document.save()
```

#### Modify Document
The dataset status and the category of a document can be modified after the document is uploaded.
To change the category, you can select the category that you desire from the project based on its ID and attribute it
to the document.

```python
category = my_project.get_category_by_id(<category_id>)

document.category = category
document.dataset_status = 2
document.save()
```

#### Delete Document
To locally delete a document, you can use:

```python
document.delete()
```

The document will be deleted from your local data folder but it will remain in the Konfuzio Server.
If you want to get it again you can update the project.
