## Prepare the data for training and testing the AI

Before training the AI model, the Documents for training and testing need to be uploaded into the Project. You can use
the Server app (here's the [tutorial](https://help.konfuzio.com/modules/documents/index.html)).

Note: all uploaded Documents have to have their Pages in the correct order.

```python
# if you want to create a new Project, use the create_new_project method
from konfuzio_sdk.api import create_new_project

project = create_new_project('project_name')
```
.. literalinclude:: /sdk/boilerplates/test_prepare_data.py
   :language: python
   :start-after: start prepare
   :end-before: end prepare
   :dedent: 4

You can also assign the status of a training or testing Document to the newly uploaded Documents as described in [Modify Document](get_started.html#modify-document) 
section: `dataset_status == 1` is equal to the training status, `dataset_status == 2` is equal to the testing status.

