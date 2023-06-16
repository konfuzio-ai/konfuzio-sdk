### Create a custom File Splitting AI

This section explains how to train a custom File Splitting AI locally, how to save it and upload it to the Konfuzio 
Server. 

By default, any [File Splitting AI](sourcecode.html#file-splitting-ai) class should derive from the 
`AbstractFileSplittingModel` class and implement the following interface:

```python
from konfuzio_sdk.api import upload_ai_model, delete_ai_model

# upload a saved model to the server
model_id = upload_ai_model(save_path)

# remove model
delete_ai_model(model_id, ai_type='file_splitting')
```

