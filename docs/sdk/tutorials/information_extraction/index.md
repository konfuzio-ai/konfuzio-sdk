.. _information-extraction-tutorials:
## Document Information Extraction

### Train a Konfuzio SDK Model to Extract Information From Payslip Documents

.. _Information Extraction:

The tutorial *RFExtractionAI Demo* aims to show you how to use the Konfuzio SDK package to use a simple `Whitespace
tokenizer <https://dev.konfuzio.com/sdk/sourcecode.html#konfuzio_sdk.tokenizer.regex.WhitespaceTokenizer>`_ and to
train a "RFExtractionAI" model to find and extract relevant information like Name, Date and Recipient
from payslip documents.

You can <a href="https://colab.research.google.com/github/konfuzio-ai/document-ai-python-sdk/blob/master/docs/sdk/tutorials/RFExtractionAI%20Demo.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> or download it from [here](https://github.com/konfuzio-ai/document-ai-python-sdk/blob/master/docs/sdk/tutorials/RFExtractionAI%20Demo.ipynb)
and try it by yourself.

.. |OpenInColab| image:: https://colab.research.google.com/assets/colab-badge.svg
.. _OpenInColab: https://colab.research.google.com/github/konfuzio-ai/document-ai-python-sdk/blob/master/docs/sdk/tutorials/RFExtractionAI%20Demo.ipynb

To prepare the data for training and testing your AI, you can follow the [data preparation tutorial](tutorials.html#tutorials.html#prepare-the-data-for-training-and-testing-the-ai).

### Train a custom Extraction AI

This section explains how to train a custom Extraction AI locally, how to save it and upload it to the Konfuzio Server. 

By default, any Extraction AI class should derive from the `AbstractExtractionAI` class and implement the following 
interface:

.. literalinclude:: /sdk/boilerplates/test_custom_extraction_ai.py
      :language: python
      :start-after: start custom
      :end-before: end custom
      :dedent: 4

Example usage of your Custom Extraction AI:

.. literalinclude:: /sdk/boilerplates/test_custom_extraction_ai.py
      :language: python
      :start-after: start init_ai
      :end-before: end init_ai
      :dedent: 4
.. literalinclude:: /sdk/boilerplates/test_custom_extraction_ai.py
      :language: python
      :start-after: start category
      :end-before: end category
      :dedent: 4
.. literalinclude:: /sdk/boilerplates/test_custom_extraction_ai.py
      :language: python
      :start-after: start train
      :end-before: end train
      :dedent: 4

The custom AI inherits from [BaseModel](sourcecode.html#base-model), which provides `BaseModel.save` to generate a 
pickle file that can be directly uploaded to the Konfuzio Server (see [Upload Extraction or Category AI to target instance](https://help.konfuzio.com/tutorials/migrate-trained-ai-to-an-new-project-to-annotate-documents-faster/index.html#upload-extraction-or-category-ai-to-target-instance)). 

Activating the uploaded AI on the web interface will enable the custom pipeline on your self-hosted installation.

It is also possible to upload the AI from your local machine using the `upload_ai_model()` method and remove it with the
`delete_ai_model()` method:

```python
from konfuzio_sdk.api import upload_ai_model, delete_ai_model

# upload a saved model to the server
model_id = upload_ai_model(pickle_model_path)

# remove model
delete_ai_model(model_id, ai_type='extraction')
```

### Example of Custom Extraction AI: The Paragraph Extraction AI

In :ref:`the Paragraph Tokenizer tutorial<paragraph-tokenizer-tutorial>`, we saw how we can use the Paragraph Tokenizer 
in `detectron` mode and with the `create_detectron_labels` option to segment a Document and create `figure`, `table`, 
`list`, `text` and `title` Annotations. The tokenizer used this way is thus able to create Annotations like in the 
following:

.. image:: /_static/img/paragraph_tokenizer.png
  :scale: 40%

Here we will see how we can use the Paragraph Tokenizer to create a Custom Extraction AI. What we need to create is 
just a simple wrapper around the Paragraph Tokenizer. It shows how you can create your own Custom Extraction AI that 
you can use in Konfuzio on-prem installations or in the [Konfuzio Marketplace](https://help.konfuzio.com/marketplace/index.html).

.. collapse:: Full Paragraph Extraction AI code

   .. literalinclude:: /sdk/boilerplates/test_paragraph_extraction_ai.py
      :language: python
      :pyobject: ParagraphExtractionAI

<br/>
Let's go step by step.

0. **Imports**

   .. literalinclude:: /sdk/boilerplates/test_paragraph_extraction_ai.py
      :language: python
      :start-after: start imports
      :end-before: end imports

1. **Custom Extraction AI model definition**

   ```python
   class ParagraphExtractionAI(AbstractExtractionAI):
   ```

   We define a class that inherits from the Konfuzio `AbstractExtractionAI` class. This class provides the interface 
   that we need to implement for our Custom Extraction AI. All Extraction AI models must inherit from this class.

2. **Add model requirements**

   .. literalinclude:: /sdk/boilerplates/test_paragraph_extraction_ai.py
      :language: python
      :start-after: start model requirements
      :end-before: end model requirements

   We need to define what the model needs to be able to run. This will inform the Konfuzio Server what information needs 
   to be made available to the model before running an extraction. If the model only needs text, you can ignore this step
   or add `requires_text = True` to make it explicit. If the model requires Page images, you will need to add 
   `requires_images = True`. Finally, in our case we also need to add `requires_segmentation = True` to inform the Server 
   that the model needs the visual segmentation information created by the Paragraph Tokenizer in `detectron` mode.

3. **Initialize the model**

   ```python
      def __init__(
         self,
         category: Category = None,
         *args,
         **kwargs,
      ):
         """Initialize ParagraphExtractionAI."""
         logger.info("Initializing ParagraphExtractionAI.")
         super().__init__(category=category, *args, **kwargs)
         self.tokenizer = ParagraphTokenizer(mode='detectron', create_detectron_labels=True)
   ```
   
   We initialize the model by calling the `__init__` method of the parent class. The only required argument is the 
   Category the Extraction AI will be used with. The Category is the Konfuzio object that contains all the Labels 
   and LabelSets that the model will use to create Annotations. This means that you need to make sure that the Category 
   object contains all the Labels and LabelSets that you need for your model. In our case, we need the `figure`, `table`, 
   `list`, `text` and `title` Labels.


4. **Define the extract method**

   .. literalinclude:: /sdk/boilerplates/test_paragraph_extraction_ai.py
      :language: python
      :pyobject: ParagraphExtractionAI.extract

   The `extract` method is the core of the Extraction AI. It takes a Document as input and returns a Document with 
   Annotations. Make sure to do a `deepcopy` of the Document that is passed so that you add the new Annotations to a 
   Virtual Document with no Annotations. The Annotations are created by the model and added to the Document. In our 
   case, we simply call the Paragraph Tokenizer in `detectron` mode and with the `create_detectron_labels` option.

5. **[OPTIONAL] Define the check_is_ready method**

   .. literalinclude:: /sdk/boilerplates/test_paragraph_extraction_ai.py
      :language: python
      :pyobject: ParagraphExtractionAI.check_is_ready

   The `check_is_ready` method is used to check if the model is ready to be used. It should return `True` if the model 
   is ready to extract, and `False` otherwise. It is checked before saving the model. You don't have to implement it, 
   and it will only check that a Category is defined. 
   
   In our case, we also check that the model contains all the Labels that we need. This is not strictly necessary, but 
   it is a good practice to make sure that the model is ready to be used.

6. **Use the model locally**

   We first make sure that all needed Labels are present in the Category.

   .. literalinclude:: /sdk/boilerplates/test_paragraph_extraction_ai.py
      :language: python
      :start-after: start create labels
      :end-before: end create labels
      :dedent: 4

   We can now use the model to extract a Document. And then we also can run extract on a Document and save the model to 
   a pickle file that can be used in Konfuzio Server.

   .. literalinclude:: /sdk/boilerplates/test_paragraph_extraction_ai.py
      :language: python
      :start-after: start use model
      :end-before: end use model
      :dedent: 4

7. **Upload the model to Konfuzio Server**
   
   You can use the Konfuzio SDK to upload your model to your on-prem installation like this:

   ```python
   from konfuzio_sdk.api import upload_ai_model

   upload_ai_model(model_path=path, category_ids=[category.id_])
   ```
   
   Once the model is uploaded you can also share your model with others on the [Konfuzio Marketplace](https://help.konfuzio.com/marketplace/index.html).

### Evaluate a Trained Extraction AI Model

In this example we will see how we can evaluate a trained `RFExtractionAI` model. We will assume that we have a trained 
pickled model available. See :ref:`here <Information Extraction>` 
for how to train such a model, and check out the [Evaluation](https://dev.konfuzio.com/sdk/sourcecode.html#ai-evaluation) 
documentation for more details.

.. literalinclude:: /sdk/boilerplates/test_evaluate_extraction_ai.py
   :language: python
   :start-after: start init
   :end-before: end init
   :dedent: 4
.. literalinclude:: /sdk/boilerplates/test_evaluate_extraction_ai.py
   :language: python
   :start-after: start scores
   :end-before: end scores
   :dedent: 4
