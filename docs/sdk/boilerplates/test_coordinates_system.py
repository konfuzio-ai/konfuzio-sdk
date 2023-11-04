"""Test code examples for coordination system documentation."""


def test_coordinates_system():
    """Test coordinates system."""
    from tests.variables import TEST_PROJECT_ID

    YOUR_PROJECT_ID = TEST_PROJECT_ID

    # Start coordinates
    from konfuzio_sdk.data import Project

    my_project = Project(id_=YOUR_PROJECT_ID)
    # first Document uploaded
    document = my_project.documents[0]
    # index of the Page to test
    page_index = 0

    width = document.pages()[page_index].width
    height = document.pages()[page_index].height
    # End coordinates
    assert width < height
