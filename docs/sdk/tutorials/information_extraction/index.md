.. _information-extraction-tutorials:
## Document Information Extraction

Information Extraction is a process of obtaining information from the Document's unstructured text and labelling it 
with Labels like Name, Date, Recipient, or any other custom Labels. The result of Extraction looks like this:

.. image:: /sdk/tutorials/information_extraction/example.png
   :width: 300px
   :align: center

### Train a custom Extraction AI

This section explains how to create a custom Extraction AI locally, how to save it and upload it to the Konfuzio Server.

Note: you don't necessarily need to create the AI from scratch if you already have some document-processing architecture.
You just need to wrap it into the class that corresponds to our Extraction AI structure. Follow the steps in this 
tutorial to find out what are the requirements for that.

To prepare the data for training or testing your AI, you can follow the [data preparation tutorial](https://dev.konfuzio.com/sdk/tutorials/data-preparation/index.html#prepare-the-data-for-training-and-testing-the-ai).

By default, any Extraction AI class should derive from the `AbstractExtractionAI` class and implement the `extract()` 
method. In this tutorial, we'll demonstrate how to create a simple custom Extraction AI that extracts dates provided in 
a certain format. Note that to enable Labels' and Label Sets' dynamic creation during extraction, you need to have
Superuser rights and enable this setting in a [Superuser Project](https://help.konfuzio.com/modules/administration/superuserprojects/index.html#create-labels-and-label-sets).

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

The custom AI inherits from AbstractExtractionAI, which in turn inherits from [BaseModel](sourcecode.html#base-model).
The inheritance can be seen on a scheme below:

.. mermaid::
    
    graph LR;
        BaseModel[BaseModel] --> AbstractExtractionAI[AbstractExtractionAI] --> CustomExtractionAI[CustomExtractionAI];

`BaseModel` provides `save` method that saves a model into a compressed pickle file that can be directly uploaded to the 
Konfuzio Server (see [Upload Extraction or Category AI to target instance](https://help.konfuzio.com/tutorials/migrate-trained-ai-to-an-new-project-to-annotate-documents-faster/index.html#upload-extraction-or-category-ai-to-target-instance)). 

Activating the uploaded AI on the web interface will enable the custom pipeline on your self-hosted installation.

Note that if you want to create Labels and Label Sets dynamically (when running the AI, instead of adding them manually
on app), you need to enable creating them in the Superuser Project settings if you have the corresponding rights.

If you have the Superuser rights, it is also possible to upload the AI from your local machine using the 
`upload_ai_model()` method and remove it with the `delete_ai_model()` method:

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
you can use in Konfuzio on-prem installations or in the [Konfuzio Marketplace](https://help.konfuzio.com/modules/marketplace/index.html).

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
   
   Once the model is uploaded you can also share your model with others on the [Konfuzio Marketplace](https://help.konfuzio.com/modules/marketplace/index.html).

### Example of Custom Extraction AI: Barcode Extraction AI with `zxing-cpp`

In this tutorial, we'll walk through the creation of a custom Barcode Extraction AI using the `zxing-cpp` library. We'll implement the Extraction AI logic, create custom Annotations, and integrate the AI with the `konfuzio-sdk` library. This AI will be able to detect barcodes from Documents and generate bounding box Annotations for the detected barcodes.

#### Requirements

---

Before we start, make sure you have the following:

1.  `Python 3.8` or higher installed. 🐍
2.  [konfuzio-sdk](https://pypi.org/project/konfuzio-sdk/) ✅
3.  The [zxing-cpp](https://github.com/zxing-cpp/zxing-cpp) library 💻

#### Step 1: Set up the `CustomAnnotation` class

---

The first step is to create a Custom [Annotation](https://dev.konfuzio.com/sdk/sourcecode.html?highlight=annotation#annotation) class that includes custom bounding boxes for our detected barcodes. These bounding boxes will be later used by the [Server](https://dev.konfuzio.com/web/index.html#what-is-the-konfuzio-server) as well as the [DVUI](https://dev.konfuzio.com/dvui/index.html#what-is-the-konfuzio-document-validation-ui) to annotate the barcodes in the Document.

```python
from typing import Dict, List
from konfuzio_sdk.data import Annotation


class CustomAnnotation(Annotation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_bboxes = kwargs.get("custom_bboxes", [])

    @property
    def bboxes(self) -> List[Dict]:
        """
        This method of the Annotation class must be overridden to return the custom_bboxes.
        :return: List of dictionaries, each in the format:
            {
                'x0': int, 'x1': int, 'y0': int, 'y1': int,
                'top': int, 'bottom': int, 'page_index': int,
                'offset_string': str, 'custom_offset_string': bool
            }
        """
        return self.custom_bboxes
```

#### Step 2: Set up the `CustomExtractionAI` class

---

The second step is to create a custom [Extraction AI](https://dev.konfuzio.com/sdk/sourcecode.html#extraction-ai) class that leverages the `CustomAnnotation` class defined in *Step 1* in order to extract barcodes from [Documents](https://dev.konfuzio.com/sdk/sourcecode.html?highlight=document#document). We will start by explaining the different methods that will be used in this class. 

**NB.** If you want to directly have the full code of the `CustomExtractionAI` class, you can skip to [Step 3](#step-3-set-up-the-barcodeextractionai-class-using-the-earlier-defined-methods).

##### Step 2.1: Extract Bounding Boxes from Image

---

In this step, we'll implement the method to extract bounding boxes and barcode text from an image using the `zxing-cpp` library.

```python
def get_bboxes_from_image(self, image, page_index):
    """
    Extract the bboxes and texts of the barcodes from an image and format the bbox dictionaries in the right format
    :param image: PIL image (the image of the page, resize to original size or probide the original image for better results )
    :param page_index: int
    """
    # import the necessary function from the library
    from zxingcpp import read_barcodes

    # create empty list to store the bboxes
    bboxes_list = []
    # get the results from the library
    barcodes_lib_results = read_barcodes(image)
    # loop through the results and extract the bboxes
    for result in barcodes_lib_results:
        # unpack the result position ## output: '496x453 743x453 743x550 496x550'
        position = str(result.position).rstrip("\x00")
        # unpack the result text ## output: '123456'
        barcode_text_value = str(result.text).rstrip("\x00")
        # unpack the coordinates of the bottom-left and top-right corners of the detected barcode
        top_right = position.split()[1].split("x")
        bottom_left = position.split()[-1].split("x")
        # create the bbox dictionary
        bbox_dict = self.get_bbox_dict_from_coords(
            top_right, bottom_left, page_index, image, barcode_text_value
        )
        # append the bbox dictionary to the list
        bboxes_list.append(bbox_dict)
    return bboxes_list
```

##### Step 2.2: Create Bbox Dictionary from `zxing-cpp` Output

---

In this step, we'll implement the method to create the Bbox dictionary from the output of `zxing-cpp`.

```python
def get_bbox_dict_from_coords(
    top_right, bottom_left, page_index, image, barcode_text_value
):
    """
    transform the coordinates of the bottom-left and top-right corners
    of the detected barcode from cv2 coordinates system to DVUI coordinates system
    """
    # get the coordinates of the top-right corner in cv2 coordinates system
    x1 = int(top_right[0])
    y1 = int(top_right[1])
    # get the coordinates of the bottom-left corner in cv2 coordinates system
    x0 = int(bottom_left[0])
    y0 = int(bottom_left[1])
    # top and bottom are resp. equal to y1 and y0 in the cv2 coordinates system
    # they don't need to be transformed to the DVUI coordinates system because they are distances and not coordinates
    top = y1
    bottom = y0
    # transform the coordinates from cv2 coordinates system to DVUI coordinates system
    # x0 and x1 don't need to be transformed because the x axis is unchanged in the DVUI coordinates system
    temp_y0 = image.height - y0
    temp_y1 = image.height - y1
    y0 = temp_y0
    y1 = temp_y1
    # create the bbox dictionary
    bbox_dict = {
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "top": top,
        "bottom": bottom,
        "page_index": page_index,
        "offset_string": barcode_text_value,
        "custom_offset_string": True,
    }
    return bbox_dict
```

With these two separate steps, you have a clear distinction between extracting bounding boxes from the image using `zxing-cpp` and creating the final bbox dictionary from the output. This makes the code more modular and easier to maintain.

##### Step 2.3: Install Dependencies

---

In this step, we'll implement the method to install the `zxing-cpp` library (since it might not be installed by default on the running environment).

```python
def install_dependencies():
    # try installing the zxing-cpp library otherwise raise an error
    try:
        import subprocess

        package_name = "zxing-cpp"
        # Run the pip install command
        subprocess.check_call(["python3.8", "-m", "pip", "install", package_name])
        print(f"The package {package_name} is ready to be used.")
    except:
        raise Exception(
            "An error occured while installing the zxing-cpp library. Please install it manually."
        )
```

##### Step 2.4: Check if the Extraction AI is ready

---

In this step, we'll implement a function that will be used to check if the Extraction AI is ready. This is needed by the server to know when to start the extraction process.

```python
def check_is_ready(self) -> bool:
    # check if the zxing-cpp library is already installed
    try:
        self.install_dependencies()
        import zxingcpp
        return True
    except:
        return False
```

#### Step 3: Set up the BarcodeExtractionAI class using the earlier defined Methods  

---

Finally, we'll create a `CustomExtractionAI` class that uses all the earlier defined functions to extract the barcode from a [Document](https://dev.konfuzio.com/sdk/sourcecode.html?highlight=document#document).

```python
from typing import Dict, List
from konfuzio_sdk.trainer.information_extraction import AbstractExtractionAI
from konfuzio_sdk.data import Document, Category, Annotation, AnnotationSet
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer
from copy import deepcopy


class CustomAnnotation(Annotation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_bboxes = kwargs.get("custom_bboxes", [])

    @property
    def bboxes(self) -> List[Dict]:
        """
        This method of the Annotation class must be overwriten to return the custom_bboxes
        :return: List of dictionaries each of the format {
            'x0': int, 'x1': int, 'y0': int, 'y1': int,
            'top': int, 'bottom': int, 'page_index': int,
            'offset_string': str, 'custom_offset_string': bool
        }
        """
        return self.custom_bboxes


class CustomExtractionAI(AbstractExtractionAI):
    """
    A Custom Extraction AI that uses the zxing-cpp library to extract barcodes from documents.
    """

    # you must set this to True if your AI requires pages images
    requires_images = True

    def __init__(self, category: Category, *args, **kwargs):
        super().__init__(category)
        self.tokenizer = WhitespaceTokenizer()

    def fit(self):
        # no training is needed since the zxing-cpp library can be used directly for extraction
        pass

    def extract(self, document: Document) -> Document:
        # check if the model is ready otherwise raise an error
        self.check_is_ready()
        result_document = super().extract(document)
        result_document._text = "this should be a long text or at least twice the number of barcodes in the document"
        barcode_label = self.project.get_label_by_name("Barcode")
        barcode_label_set = self.project.get_label_set_by_name("Barcodes Set")
        barcode_annotation_set = AnnotationSet(
            document=result_document, label_set=barcode_label_set
        )
        # loop through the pages of the document and extract the barcodes
        for page_index, page in enumerate(document.pages()):
            page_width = page.width
            page_height = page.height
            # get the page in image format
            image = page.get_image(update=True)
            # convert the image to RGB
            image = image.convert("RGB")
            # resize the image to the page size
            # IMPORTANT: since the image is already resized we won't need any scaling factors to transform the coordinates
            image = image.resize((int(page_width), int(page_height)))
            # get the bboxes and texts of the barcodes using zxing-cpp
            page_bboxes_list = self.get_bboxes_from_image(image, page_index)
            # loop through the bboxes and create the annotations using enumerate
            for bbox_index, bbox_dict in enumerate(page_bboxes_list):
                _ = CustomAnnotation(
                    document=result_document,
                    annotation_set=barcode_annotation_set,
                    spans=[],
                    start_offset=bbox_index + 1,
                    end_offset=bbox_index + 2,
                    label=barcode_label,
                    label_set=barcode_label_set,
                    confidence=1.0,
                    bboxes=None,
                    custom_bboxes=[bbox_dict],
                )

        return result_document

    def get_bboxes_from_image(self, image, page_index):
        from zxingcpp import read_barcodes
        bboxes_list = []
        barcodes_lib_results = read_barcodes(image)
        for result in barcodes_lib_results:
            position = str(result.position).rstrip("\x00")
            barcode_text_value = str(result.text).rstrip("\x00")
            top_right = position.split()[1].split("x")
            bottom_left = position.split()[-1].split("x")
            bbox_dict = self.get_bbox_dict_from_coords(
                top_right, bottom_left, page_index, image, barcode_text_value
            )
            bboxes_list.append(bbox_dict)
        return bboxes_list

    def get_bbox_dict_from_coords(
        self, top_right, bottom_left, page_index, image, barcode_text_value
    ):
        x1 = int(top_right[0])
        y1 = int(top_right[1])
        x0 = int(bottom_left[0])
        y0 = int(bottom_left[1])
        top = y1
        bottom = y0
        temp_y0 = image.height - y0
        temp_y1 = image.height - y1
        y0 = temp_y0
        y1 = temp_y1
        bbox_dict = {
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "top": top,
            "bottom": bottom,
            "page_index": page_index,
            "offset_string": barcode_text_value,
            "custom_offset_string": True,
        }
        return bbox_dict

    def install_dependencies(self):
        try:
            import subprocess
            package_name = "zxing-cpp"
            subprocess.check_call(
                ["pip", "install", package_name])
            print(f"The package {package_name} is ready to be used.")
        except:
            raise Exception(
                "An error occured while installing the zxing-cpp library. Please install it manually."
            )

    def check_is_ready(self) -> bool:
        try:
            self.install_dependencies()
            import zxingcpp
            return True
        except:
            return False
```

#### Step 4: Putting It All Together

---

Now, let's create the main script to run the custom Extraction AI and process the Documents:

We start by defining our `project_id` (don't forget to change the project_id to your own project_id, it should be an `int`) then we save the Extraction AI as a [pickle](https://docs.python.org/3/library/pickle.html) file that we will upload to the [Server](https://dev.konfuzio.com/web/index.html#what-is-the-konfuzio-server) later.

```python
project_id = "my_project_id"
```

```python
from konfuzio_sdk.data import Project

project = Project(id_=project_id, update=True, strict_data_validation=False)
barcode_extraction_ai = CustomExtractionAI(category=project.categories[0])
pickle_model_path = barcode_extraction_ai.save()
```

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