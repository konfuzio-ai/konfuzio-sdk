"""Test code examples for Get Started section of the documentation."""
from konfuzio_sdk.data import Project, Document

YOUR_PROJECT_ID = 46
FILE_PATH = 'docs/sdk/boilerplates/pdf.pdf'
ASSIGNEE_ID = 1043

my_project = Project(id_=YOUR_PROJECT_ID)
document = Document.from_file_sync(FILE_PATH, project=my_project)
my_project.get(update=True)
my_project = Project(id_=YOUR_PROJECT_ID, update=True)

documents = my_project.documents
test_documents = my_project.test_documents

for document in my_project.documents:
    document.get_file()

for document in my_project.documents:
    document.get_file(ocr_version=False)

for document in my_project.documents:
    document.get_images()

for document in my_project.documents:
    document.get_bbox()

my_project.documents_folder
document.update()
document.delete()
document = Document.from_file_sync(FILE_PATH, project=my_project)

my_project.init_or_update_document(from_online=False)
document_id = document.id_

document = my_project.get_document_by_id(document_id)

document = my_project.documents[0]
document.assignee = ASSIGNEE_ID

document.save_meta_data()
