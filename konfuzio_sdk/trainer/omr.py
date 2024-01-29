"""

"""


from konfuzio_sdk.extras import Module
import abc
from typing import Dict, List, Tuple, Union
from konfuzio_sdk.data import Document

class OMRAbstractModel(Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__()

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, FloatTensor]:
        """Forward pass of the model.

        Args:
            input: Dictionary containing the input tensors.

        Returns:
            Dictionary containing the output tensors.
        """
        pass


class CheckboxDetector(OMRAbstractModel):
        def __init__(
        self,*
        path: str,
        **kwargs,
    ):
        """Init and set parameters."""
        super().__init__()
        # Load the checkbox detection model using it's path from where it's hosted
        # For the moment, it will be hosted in the Konfuzio S3 bucket 
        # checkboxes = torch.load_model(path)(page_image)

    def forward(self, page_image: Tensor) -> Dict[str, FloatTensor]:
        """Forward pass of the model.

        Args:
            input: Dictionary containing the input tensors. # TBD !TODO

        Returns:
            Dictionary containing the output tensors. # TBD !TODO
        """
        # checkboxes = {
        #       "class": "checked" / "unchecked",
        #       "bbox": [x1, y1, x2, y2],
        # } 

        # return checkboxes
        pass


def map_annotations_to_checkboxes(document: Document) -> Document:
    """Map the annotations to the checkboxes.

    Args:
        document: Document object containing the annotations and the page image.

    Returns:
        Document object containing the annotations and the page image with the mapped checkboxes.
    """
    # Get the page image
    page_image = document.page_image

    # Get the annotations
    annotations = document.annotations # TBD !TODO

    # Get the checkboxes
    checkboxes = CheckboxDetector(page_image)

    # Map the annotations to the checkboxes
    # Loop through the annotations, check if the Label has label.is_linked_to_checkbox == True.
    # If so, apply the logic of finding the closest checkbox to the annotation
    # TBD !TODO
    # Suppose we found the closest checkbox to the annotation, then we save it in Annotation.metadata
    # # annotation.metadata["omr"]["is_checked"] = checkbox["class"]
    # # annotation.metadata["omr"]["checkbox_bbox"] = checkbox["bbox"]

    # Return the document
    return document


