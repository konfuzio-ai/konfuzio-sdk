"""Test code examples for Get Started section of the documentation."""
import os

from konfuzio_sdk.data import Project, Document

YOUR_PROJECT_ID = 46
FILE_PATH = 'docs/sdk/boilerplates/pdf.pdf'
ASSIGNEE_ID = None

my_project = Project(id_=YOUR_PROJECT_ID, update=True)
assert len(my_project._documents) == 82
my_project.get(update=True)
my_project = Project(id_=YOUR_PROJECT_ID, update=True)

documents = my_project.documents
test_documents = my_project.test_documents
assert len(documents) == 26
assert len(test_documents) == 3

for document in my_project.documents:
    document.get_file()

for document in my_project.documents:
    document.get_file(ocr_version=False)
    assert os.path.exists(document.file_path)

for document in my_project.documents:
    document.get_images()
    for page in document.pages():
        assert os.path.exists(page.image_path)

for document in my_project.documents:
    document.get_bbox()
    assert os.path.exists(document.bbox_file_path)

assert os.path.exists(my_project.documents_folder)
my_project.documents_folder
document = my_project.documents[0]
document.update()
document.delete()
my_project = Project(id_=46, update=True)

document = Document.from_file_sync(FILE_PATH, project=my_project)
document = my_project._documents[-1]
document.delete(delete_online=True)
document = Document.from_file_async(FILE_PATH, project=my_project)
document = my_project._documents[-1]
document.dataset_status = 0

my_project.init_or_update_document(from_online=False)
document_id = document.id_
document = my_project.get_document_by_id(document_id)
document.delete(delete_online=True)
document = my_project.documents[0]
document.assignee = ASSIGNEE_ID
document.dataset_status = 2

document.save_meta_data()
my_project = Project(id_=YOUR_PROJECT_ID, update=True)
assert len(my_project.documents) == 26
