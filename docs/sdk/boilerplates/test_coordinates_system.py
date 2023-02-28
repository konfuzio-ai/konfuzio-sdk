"""Test code examples for coordination system documentation."""
from konfuzio_sdk.data import Project

from variables import YOUR_PROJECT_ID

my_project = Project(id_=YOUR_PROJECT_ID)
# first Document uploaded
document = my_project.documents[0]
# index of the Page to test
page_index = 0

width = document.pages()[page_index].width
height = document.pages()[page_index].height
assert width < height
