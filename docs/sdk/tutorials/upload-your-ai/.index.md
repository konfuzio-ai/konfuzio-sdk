## Upload your AI

If you want to upload a model that you've built locally to the Server, you can use one of the two options, provided 
that you have the Superuser rights.

First option is manual, using the steps from the [tutorial](https://help.konfuzio.com/tutorials/migrate-trained-ai-to-an-new-project-to-annotate-documents-faster/index.html#upload-extraction-or-category-ai-to-target-instance).

Second is using the method `upload_ai_model()` from `konfuzio_sdk.api`. Arguments are different for different types of 
AI. The method returns the model's ID that can later be used to update or delete the model.

```python
from konfuzio_sdk.api import upload_ai_model

# upload a saved model to the server
extraction_model_id = upload_ai_model(pickle_model_path, ai_type='extraction', category_id=YOUR_CATEGORY_ID)
categorization_model_id = upload_ai_model(pickle_model_path, ai_type='categorization', project_id=YOUR_PROJECT_ID)
splitting_model_id = upload_ai_model(pickle_model_path, ai_type='filesplitting', project_id=YOUR_PROJECT_ID)
```

You can update an uploaded model via the `update_ai_model()` method. The information you can change is model's name and 
description.

```python
from konfuzio_sdk.api import update_ai_model

# update the uploaded model – the ai_type is different for different AIs
update_ai_model(YOUR_MODEL_ID, ai_type='extraction')
update_ai_model(YOUR_MODEL_ID, ai_type='categorization')
update_ai_model(YOUR_MODEL_ID, ai_type='filesplitting')
```

You can also remove an uploaded model by using `delete_ai_model()`.

```python
from konfuzio_sdk.api import delete_ai_model

# delete the uploaded model – the ai_type is different for different AIs
delete_ai_model(YOUR_MODEL_ID, ai_type='extraction')
delete_ai_model(YOUR_MODEL_ID, ai_type='categorization')
delete_ai_model(YOUR_MODEL_ID, ai_type='filesplitting')
```