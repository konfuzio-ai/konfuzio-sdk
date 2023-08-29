.. \_barcode-scanner-tutorials:

## Barcode Scanner with zxing-cpp

In this tutorial, we'll walk through the creation of a custom Barcode Extraction AI using the `zxing-cpp` library. We'll implement the Extraction AI logic, create custom Annotations, and integrate the AI with the `konfuzio-sdk` library. This AI will be able to detect barcodes from Documents and generate bounding box Annotations for the detected barcodes.

### Requirements

---

Before we start, please ensure that the following requirements are available and properly installed on your system:

1.  `Python 3.8` or a higher version. 🐍
2.  The [konfuzio-sdk](https://pypi.org/project/konfuzio-sdk/) package. ✅
3.  The [zxing-cpp](https://github.com/zxing-cpp/zxing-cpp) library. 💻

### 1. Set up the `CustomAnnotation` class

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

### 2. Defining the `CustomExtractionAI` class methods

---

The second step is to create a custom [Extraction AI](https://dev.konfuzio.com/sdk/sourcecode.html#extraction-ai) class that leverages the `CustomAnnotation` class defined in _Step 1_ in order to extract barcodes from [Documents](https://dev.konfuzio.com/sdk/sourcecode.html?highlight=document#document). We will start by explaining the different methods that will be used in this class.

**NB.** If you want to directly have the full code of the `CustomExtractionAI` class, you can skip to [Step 3](#putting-it-all-together).

#### 2.1. Extract Bounding Boxes from Image

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

#### 2.2. Create Bbox Dictionary from `zxing-cpp` Output

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

#### 2.3. Install Dependencies

---

In this step, we'll implement the method to install the `zxing-cpp` library (since it might not be installed by default on the running environment).

```python
def install_dependencies():
    # try installing the zxing-cpp library otherwise raise an error
    try:
        import subprocess

        package_name = "zxing-cpp"
        # Run the pip install command
        subprocess.check_call(["pip", "install", package_name])
        print(f"The package {package_name} is ready to be used.")
    except:
        raise Exception(
            "An error occured while installing the zxing-cpp library. Please install it manually."
        )
```

#### 2.4. Check if the Extraction AI is ready

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

### 3. Putting it All Together

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

### 4. Saving the `CustomExtractionAI`

---

Now, let's create the main script to save the custom Extraction AI that will be used to process Documents on Konfuzio:

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

This is an example of how the output of this Barcode Scanner will look like on the [DVUI](https://dev.konfuzio.com/dvui/index.html#what-is-the-konfuzio-document-validation-ui):

.. image:: /sdk/tutorials/barcode-scanner/barcode_scanner_example.png
