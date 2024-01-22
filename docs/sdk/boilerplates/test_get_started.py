"""Test code examples for Get Started section of the documentation."""
import os
import time

from konfuzio_sdk.api import delete_file_konfuzio_api
from konfuzio_sdk.data import Document, Project

FILE_PATH = 'tests/test_data/pdf.pdf'
ASSIGNEE_ID = None


def test_project():
    """Test initializing the Project."""
    from tests.variables import TEST_PROJECT_ID

    YOUR_PROJECT_ID = TEST_PROJECT_ID
    # start project
    my_project = Project(id_=YOUR_PROJECT_ID)
    # end project
    # start get
    my_project.get(update=True)
    # end get
    # start update
    my_project = Project(id_=YOUR_PROJECT_ID, update=True)
    # end update


def test_documents():
    """Test Document initialization."""
    from tests.variables import TEST_PROJECT_ID

    YOUR_PROJECT_ID = TEST_PROJECT_ID
    my_project = Project(id_=YOUR_PROJECT_ID)

    # start documents
    documents = my_project.documents
    # end documents
    # start test_documents
    test_documents = my_project.test_documents
    # end test_documents
    assert len(documents) == 26
    assert len(test_documents) == 3


def test_document_loading():
    """Test uploading Documents."""
    from tests.variables import TEST_PROJECT_ID

    YOUR_PROJECT_ID = TEST_PROJECT_ID
    my_project = Project(id_=YOUR_PROJECT_ID)
    # start sync_true
    document = Document.from_file(FILE_PATH, project=my_project, sync=True)
    # end sync_true
    document = my_project._documents[-1]
    document.dataset_status = 0
    # start online_delete
    document.delete(delete_online=True)
    # end online_delete
    # start sync_false
    document = Document.from_file(FILE_PATH, project=my_project, sync=False)
    # end sync_false
    # start pulling loop
    for i in range(5):
        document.update()
        if document.ocr_ready is True:
            print(document.text)
            break
        time.sleep(i * 10 + 3)
        # end pulling loop
        break

    document.delete(delete_online=True)

    # start timeout upload
    document = Document.from_file(FILE_PATH, project=my_project, timeout=300, sync=True)  # 300 seconds timeout
    # end timeout upload

    # start update_doc
    document.update()
    # end update_doc
    document_id = document.id_
    # start delete
    document.delete()
    # end delete
    delete_file_konfuzio_api(document_id)


def test_modify_document():
    """Test modifying the Document."""
    from tests.variables import TEST_PROJECT_ID

    YOUR_PROJECT_ID = TEST_PROJECT_ID
    my_project = Project(id_=YOUR_PROJECT_ID)
    document = my_project.documents[0]
    # start assignee
    document.assignee = ASSIGNEE_ID
    document.dataset_status = 2

    document.save_meta_data()
    # end assignee

    # start get file
    for document in my_project.documents:
        document.get_file()
    # end get file
    # start get original
    for document in my_project.documents:
        document.update()
        document.get_file(ocr_version=False)
        # end get original
        assert os.path.exists(document.file_path)
    # start get images
    for document in my_project.documents:
        document.get_images()
        # end get images
        for page in document.pages():
            assert os.path.exists(page.image_path)
    # start get bbox
    for document in my_project.documents:
        document.get_bbox()
        # end get bbox
        assert os.path.exists(document.bbox_file_path)
    assert os.path.exists(my_project.documents_folder)
    # start folder
    my_project.documents_folder
    # end folder
    # start document text path
    document.txt_file_path
    # end document text path
    # start document bbox path
    document.bbox_file_path
    # end document bbox path
