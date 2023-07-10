"""Test code examples for the tutorial on data preparation for the AI training and testing."""

FILE_PATH = 'tests/test_data/pdf.pdf'


def test_prepare_data():
    """Upload Documents sychronously and ensure they have Categories."""
    from tests.variables import TEST_PROJECT_ID

    FILE_PATH_1 = FILE_PATH_2 = FILE_PATH_3 = FILE_PATH

    # start prepare
    # if you want to use the existing Project, initialize it
    from konfuzio_sdk.data import Project, Document

    project = Project(id_=TEST_PROJECT_ID)

    # provide paths to your locally stored Documents
    file_paths = [FILE_PATH_1, FILE_PATH_2, FILE_PATH_3]
    for document_path in file_paths:
        # create new Documents from your local files
        _ = Document.from_file(document_path, project=project, sync=False)
        # end prepare
        _.delete(delete_online=True)
