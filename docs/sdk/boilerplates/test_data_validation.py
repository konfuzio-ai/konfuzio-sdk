"""Test Data Validation code examples from the documentation."""
from copy import deepcopy

from konfuzio_sdk.data import Project

YOUR_PROJECT_ID = 46

project = Project(id_=YOUR_PROJECT_ID)
assert len(project.documents) == 26
project = Project(id_=YOUR_PROJECT_ID, strict_data_validation=False)
doc = project.documents[0]
virtual_doc = deepcopy(doc)
assert virtual_doc.bboxes
virtual_doc.set_text_bbox_hashes()
virtual_doc._text = '123' + doc.text  # Change text to bring bbox out of sync.
virtual_doc.check_bbox()
