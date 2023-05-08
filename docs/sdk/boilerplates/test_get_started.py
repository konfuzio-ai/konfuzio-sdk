"""Test code examples for Get Started section of the documentation."""
import os

from konfuzio_sdk.data import Project, Document

from tests.variables import TEST_PROJECT_ID

YOUR_PROJECT_ID = TEST_PROJECT_ID
FILE_PATH = 'tests/test_data/pdf.pdf'
ASSIGNEE_ID = None


def test_get_started():
    """Test Get Started section of the documentation."""
    my_project = Project(id_=YOUR_PROJECT_ID)

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

    document = Document.from_file(FILE_PATH, project=my_project, sync=True)
    document = my_project._documents[-1]
    document.dataset_status = 0
    document.delete(delete_online=True)
    document = Document.from_file(FILE_PATH, project=my_project, sync=False)
    document.update()

    document_id = document.id_
    document = my_project.get_document_by_id(document_id)

    document = my_project.documents[0]
    document.assignee = ASSIGNEE_ID
    document.dataset_status = 2

    document.save_meta_data()
    my_project = Project(id_=YOUR_PROJECT_ID, update=True)
    assert len(my_project.documents) == 26
    my_project.get_document_by_id(document_id).delete(delete_online=True)

    def test_placeholder():
        """Placeholder for pytest."""
        pass
