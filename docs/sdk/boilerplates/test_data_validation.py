"""Test Data Validation code examples from the documentation."""


def test_data_validation():
    """Test data validation."""
    from copy import deepcopy
    from tests.variables import TEST_PROJECT_ID

    YOUR_PROJECT_ID = TEST_PROJECT_ID

    # Start initialization
    from konfuzio_sdk.data import Project

    project = Project(id_=YOUR_PROJECT_ID)  # all the data in this Project will be validated
    # End initialization
    assert len(project.documents) == 26
    # Start no val
    project = Project(id_=YOUR_PROJECT_ID, strict_data_validation=False)
    # End no val
    doc = project.documents[0]
    virtual_doc = deepcopy(doc)
    assert virtual_doc.bboxes
    virtual_doc.set_text_bbox_hashes()
    virtual_doc._text = '123' + doc.text
    virtual_doc.check_bbox()
