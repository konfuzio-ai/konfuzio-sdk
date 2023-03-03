"""Test Data Validation code examples from the documentation."""
from copy import deepcopy

from konfuzio_sdk.data import Project

from variables import YOUR_PROJECT_ID

project = Project(id_=YOUR_PROJECT_ID)  # all the data in this Project will be validated
assert len(project.documents) == 26
project = Project(id_=YOUR_PROJECT_ID, strict_data_validation=False)
doc = project.documents[0]
virtual_doc = deepcopy(doc)
assert virtual_doc.bboxes
virtual_doc.set_text_bbox_hashes()
virtual_doc._text = '123' + doc.text
virtual_doc.check_bbox()
