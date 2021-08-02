## Example Usage

### Project

Retrieve all information available for your project:

```python
my_project = Project()
```

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

None: 0  
Preparation: 1  
Training: 2  
Test: 3  
Low OCR Quality: 4  

For example, to get all documents in the project, you can do:

```python
documents = my_project.get_documents_by_status(dataset_statuses=[0, 1, 2, 3, 4])
```

The training and test documents can be accessed directly by:

```python
training_documents = my_project.documents
test_documents = my_project.test_documents
```

By default, you get 4 files for each document that contain information of the text, pages, sections and annotations.

document.txt  
pages.json5  
sections.json5  
annotations.json5  

To get the pdfs of the documents, you can use **get_file()**.

```python
for document in my_project.documents:
    document.get_file()
```

This will download the sandwich file which, besides the document itself, it contains the bounding boxes information of
the characters.

In the document folder, you will see a new file with the original name followed by "_ocr".

To get the pages of the document as png images, you can use **get_images()**.

```python
for document in my_project.documents:
    document.get_images()
```

You will get one png image named "page_<number_of_page>.png" for each page in the document.

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

If there are changes in the document in the Konfuzio App, you can update the document with:

```python
document.update()
```
