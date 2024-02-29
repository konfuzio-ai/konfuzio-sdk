"""Optical Mark Recognition (OMR) for processing checkboxes."""

import abc
import logging
from pathlib import Path
from typing import Any, List, Tuple, Union

import bentoml
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.optimize import linear_sum_assignment

from konfuzio_sdk.data import Bbox, Document
from konfuzio_sdk.extras import Module

logger = logging.getLogger(__name__)

# Define a type alias for clarity
Box = Union[List[Tuple[int, int, int, int]], np.ndarray]


class OMRAbstractModel(Module, metaclass=abc.ABCMeta):
    """Abstract class for the Optical Mark Recognition model."""

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize the OMRAbstractModel.

        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.requires_text = True
        self.requires_images = True
        self.name = self.__class__.__name__

    # @abc.abstractmethod
    # def forward(self, image: Image.Image) -> Any:
    #    """
    #    Abstract method to be implemented by subclasses. Defines the forward pass of the model.

    #    :param image: The input image to process.
    #    :type image: PIL.Image.Image
    #    :return: The result of the model prediction.
    #    :rtype: Any
    #    """

    @property
    def bento_metadata(self) -> dict:
        """Metadata to include into the bento-saved instance of a model."""
        return {'requires_images': self.requires_images, 'requires_segmentation': self.requires_text}

    @property
    def entrypoint_methods(self) -> dict:
        """Methods that will be exposed in a bento-saved instance of a model."""
        return {
            '__call__': {'batchable': False},
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        # Modify state to be serializable, if necessary
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore any necessary state not directly saved in __getstate__

    def save_bento(self, build=True, output_dir=None) -> Union[None, tuple]:
        """
        Save AI as a BentoML model in the local store.

        :param build: Bundle the model into a BentoML service and store it in the local store.
        :param output_dir: If present, a .bento archive will also be saved to this directory.

        :return: None if build=False, otherwise a tuple of (saved_bento, archive_path).
        """
        if output_dir and not build:
            raise ValueError('Cannot specify output_dir without build=True')

        custom_objects = {
            'utils': CheckboxDetector,
        }

        model_path = str(Path(__file__).parent / 'ckpt_best.pt')
        if not Path(model_path).exists():
            raise FileNotFoundError(f'Model weights file not found at {model_path}')
        detector = torch.jit.load(model_path)

        saved_model = bentoml.torchscript.save_model(
            name=self.name.lower(),
            model=detector,
            signatures=self.entrypoint_methods,
            metadata=self.bento_metadata,
            custom_objects=custom_objects,
        )
        logger.info(f'Model saved in the local BentoML store: {saved_model}')

        if not build:
            return

        saved_bento = self.build_bento(bento_model=saved_model)
        logger.info(f'Bento created: {saved_bento}')

        if not output_dir:
            return saved_bento, None  # None = no archive saved

        archive_path = saved_bento.export(output_dir)
        logger.info(f'Bento archive saved: {archive_path}')

        return saved_bento, archive_path

    # is this function even needed? TODO: check
    def load_bento(model_name):
        """Load AI as a Bento ML instance."""
        return bentoml.torchscript.load_model(model_name)

    def build_bento(self, bento_model):
        # Build BentoML service for the model.
        return bentoml.bentos.build(
            name=self.name.lower(),
            service=f'{self.name.lower()}_service.py:svc',
            include=[f'{self.name.lower()}_service.py', 'schemas.py'],
            python={'packages': ['konfuzio_sdk[ai]'], 'lock_packages': True},
            build_ctx=Path(__file__).resolve().parent.parent
            / 'bento'
            / 'extraction',  # os.path.dirname(os.path.abspath(__file__)) + '/../bento/extraction',
            models=[str(bento_model.tag)],
        )


class CheckboxDetector(OMRAbstractModel):
    """Detect checkboxes in images using a pre-trained model."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the CheckboxDetector with a pre-trained model and default parameters.

        :param kwargs: Arbitrary keyword arguments.
        :type kwargs: :class:`Any`
        :raises FileNotFoundError: If the model weights file is not found.
        :return: None
        :rtype: :class:`NoneType`
        """
        super().__init__()
        # model_path = str(Path(__file__).parent / 'ckpt_best.pt')
        # if not Path(model_path).exists():
        #    raise FileNotFoundError(f'Model weights file not found at {model_path}')
        # self.detector = torch.jit.load(model_path)
        self.input_shape = (1280, 1280)
        self.threshold = 0.7

    def _threshold(self, cls_conf: np.ndarray, bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filters detections based on confidence threshold.

        :param cls_conf: Confidence scores for each detection.
        :type cls_conf: :class:`numpy.ndarray`
        :param bboxes: Bounding boxes for each detection.
        :type bboxes: :class:`numpy.ndarray`
        :return: Filtered confidence scores and bounding boxes.
        :rtype: :class:`tuple`
        """
        idx = np.argwhere(cls_conf > self.threshold)
        cls_conf = cls_conf[idx[:, 0]]
        bboxes = bboxes[idx[:, 0]]
        return cls_conf, bboxes

    def _nms(self, cls_conf: np.ndarray, bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies Non-Maximum Suppression to detections.

        :param cls_conf: Confidence scores for each detection.
        :type cls_conf: :class:`numpy.ndarray`
        :param bboxes: Bounding boxes for each detection.
        :type bboxes: :class:`numpy.ndarray`
        :return: Detections after applying NMS.
        :rtype: :class:`tuple`
        """
        indices = torchvision.ops.nms(
            torch.from_numpy(bboxes), torch.from_numpy(cls_conf.max(1)), iou_threshold=0.5
        ).numpy()
        cls_conf = cls_conf[indices]
        bboxes = bboxes[indices]
        return cls_conf, bboxes

    def _rescale(self, image: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
        """
        Rescales an image to a specified output shape.

        :param image: The input image.
        :type image: :class:`numpy.ndarray`
        :param output_shape: Desired output shape.
        :type output_shape: :class:`tuple`
        :return: Rescaled image.
        :rtype: :class:`numpy.ndarray`
        """
        height, width = image.shape[:2]
        scale_factor = min(output_shape[0] / height, output_shape[1] / width)
        if scale_factor != 1.0:
            new_height, new_width = round(height * scale_factor), round(width * scale_factor)
            image = Image.fromarray(image)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            image = np.array(image)
        return image

    def _bottom_right_pad(
        self, image: np.ndarray, output_shape: Tuple[int, int], pad_value: Tuple[int, int, int] = (114, 114, 114)
    ) -> np.ndarray:
        """
        Pads the image on the bottom and right to reach the output shape.

        :param image: The input image.
        :type image: :class:`numpy.ndarray`
        :param output_shape: Desired output shape.
        :type output_shape: :class:`tuple`
        :param pad_value: Padding value.
        :type pad_value: :class:`tuple`
        :return: Padded image.
        :rtype: :class:`numpy.ndarray`
        """

        height, width = image.shape[:2]
        pad_height = output_shape[0] - height
        pad_width = output_shape[1] - width

        pad_h = (0, pad_height)  # top=0, bottom=pad_height
        pad_w = (0, pad_width)  # left=0, right=pad_width

        constant_values = ((pad_value, pad_value), (pad_value, pad_value), (0, 0))
        # Fixes issue with numpy deprecation warning since constant_values is ragged array (Have to explicitly specify object dtype)
        constant_values = np.array(constant_values, dtype=np.object_)

        padding_values = (pad_h, pad_w, (0, 0))
        processed_image = np.pad(image, pad_width=padding_values, mode='constant', constant_values=constant_values)

        return processed_image

    def _permute(self, image: np.ndarray, permutation: Tuple[int, int, int] = (2, 0, 1)) -> np.ndarray:
        """
        Permutes the image channels.

        :param image: The input image.
        :type image: :class:`numpy.ndarray`
        :param permutation: Channel permutation.
        :type permutation: :class:`tuple`
        :return: Permuted image.
        :rtype: :class:`numpy.ndarray`
        """
        processed_image = np.ascontiguousarray(image.transpose(permutation))
        return processed_image

    def _standardize(self, image: np.ndarray, max_value=255) -> np.ndarray:
        """
        Standardizes the pixel values of the image.

        :param image: The input image.
        :type image: :class:`numpy.ndarray`
        :param max_value: The maximum pixel value.
        :type max_value: :class:`int`
        :return: Standardized image.
        :rtype: :class:`numpy.ndarray`
        """
        processed_image = (image / max_value).astype(np.float32)
        return processed_image

    def _preprocess(self, image: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
        """
        Preprocesses the image before passing it to the model.
        The preprocessing is done as during training, so do not change anything.

        :param image: The input image.
        :type image: :class:`numpy.ndarray`
        :param out_shape: Desired output shape.
        :type out_shape: :class:`tuple`
        :return: Preprocessed image.
        :rtype: :class:`numpy.ndarray`
        """
        logger.info('Run checkbox detection preprocessing.')
        if image.mode == 'P':
            image = image.convert('RGB')
        image = np.array(image)[:, :, ::-1]  # convert to np and BGR as during training
        image = self._rescale(image, output_shape=out_shape)
        image = self._bottom_right_pad(image, output_shape=out_shape, pad_value=(114, 114, 114))
        image = self._permute(image, permutation=(2, 0, 1))
        image = self._standardize(image, max_value=255)
        image = image[np.newaxis, ...]  # add batch dimension
        image = torch.from_numpy(image)
        return image

    def _postprocess(self, outputs: torch.Tensor, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocesses the model's outputs to obtain final detections.

        :param outputs: Model's raw outputs.
        :type outputs: :class:`torch.Tensor`
        :param image_shape: Original image shape.
        :type image_shape: :class:`tuple`
        :return: Confidence scores and bounding boxes after postprocessing.
        :rtype: :class:`tuple`
        """
        logger.info('Run checkbox detection postprocessing.')
        cls_conf = outputs[1][0].cpu().detach().numpy()
        bboxes = outputs[0][0].cpu().detach().numpy()
        cls_conf, bboxes = self._threshold(cls_conf, bboxes)
        cls_conf, bboxes = self._nms(cls_conf, bboxes)

        # Define and apply scale for the bounding boxes to the original image size
        scaler = max((image_shape[1] / self.input_shape[1], image_shape[0] / self.input_shape[0]))
        bboxes *= scaler
        bboxes = np.array([(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in bboxes])
        return cls_conf, bboxes

    # def forward(self, image: Image.Image) -> Tuple[np.ndarray, List[bool], np.ndarray]:
    #    """
    #    Runs the detection pipeline on an input image.

    #    :param image: The input image.
    #    :type image: :class:`PIL.Image.Image`
    #    :return: Bounding boxes, detection flags, and confidence scores.
    #    :rtype: :class:`tuple`
    #    """
    #    logger.info('Run checkbox detection.')
    #    input_image = self._preprocess(image, self.input_shape)
    #    outputs = self.detector(input_image)
    #    cls_conf, bboxes = self._postprocess(outputs, image.size)
    #    checked = [True if c[0] > c[1] else False for c in cls_conf]
    #    return bboxes, checked, cls_conf


class BboxPairing:
    """
    Pair two sets of bounding boxes, ensuring the closest bounding boxes are matched together.

    The algorithm calculates the minimum edge-to-edge distance between all pairs of boxes from two classes.
    It then uses the Hungarian algorithm to find the optimal pairing of boxes that minimizes the total edge-to-edge distance.
    The points used to calculate the distances are the middle points of the edges of the bounding boxes.

    In case one set of boxes has more elements than the other, the algorithm will pair the closest boxes and leave the remaining ones unpaired.

    The following example shows how you can use the `BboxPairing` class to find pairs.

    .. testcode::

        class1_boxes = [(0, 0, 1, 1), (10, 10, 12, 12)]
        class2_boxes = [(2, 0, 3, 1), (14, 14, 16, 16)]
        bbox_pairing = BboxPairing()
        class1_ind, class2_ind = bbox_pairing.find_pairs(class1_boxes, class2_boxes)

        for i, j in zip(class1_ind, class2_ind):
            print(f"Class 1 Box {i} is paired with Class 2 Box {j}.")

    .. testoutput::

        Class 1 Box 0 is paired with Class 2 Box 0.
        Class 1 Box 1 is paired with Class 2 Box 1.

    """

    def _mid_points(self, boxes: Box) -> np.ndarray:
        """
        For each box, calculate the middle points of its edges.

        :param boxes: A list of bounding boxes, each defined as a tuple in format (x0, y0, x1, y1).
        :type boxes: list[tuple[float, float, float, float]]
        :return: A list of arrays, with each array containing four middle points (top, bottom, left, right) for each box.
        :rtype: list[np.ndarray]
        """
        middle_points = np.zeros((len(boxes), 4, 2))  # 4 edges, 2 coordinates (x, y) each
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            middle_points[i] = [
                [(x1 + x2) / 2, y1],  # Top
                [(x1 + x2) / 2, y2],  # Bottom
                [x1, (y1 + y2) / 2],  # Left
                [x2, (y1 + y2) / 2],  # Right
            ]
        return middle_points

    def _pair_distances(self, middle_points1: np.ndarray, middle_points2: np.ndarray) -> np.ndarray:
        """
        Calculate distances between all pairs of middle points from two sets of bounding boxes.
        `middle_points1` and `middle_points2` are arrays of middle points precomputed.

        :param middle_points1: Array of middle points for the first set of boxes.
        :type middle_points1: np.ndarray
        :param middle_points2: Array of middle points for the second set of boxes.
        :type middle_points2: np.ndarray
        :return: Distance matrix between all pairs of points.
        :rtype: np.ndarray
        """
        # Expand dimensions to enable broadcasting: (n_points1, 1, 2) and (1, n_points2, 2)
        expanded_points1 = np.expand_dims(middle_points1, 1)
        expanded_points2 = np.expand_dims(middle_points2, 0)

        # Compute distances using broadcasting
        distances = np.sqrt(np.sum((expanded_points1 - expanded_points2) ** 2, axis=2))
        return distances

    def _min_edge_distances(self, class1_boxes: Box, class2_boxes: Box) -> np.ndarray:
        """
        Calculate the minimum edge-to-edge distance between boxes in class 1 and class 2.

        :param class1_boxes: Bounding boxes of the first class in format (x0, y0, x1, y1).
        :type class1_boxes: list[tuple[float, float, float, float]]
        :param class2_boxes: Bounding boxes of the second class in format (x0, y0, x1, y1).
        :type class2_boxes: list[tuple[float, float, float, float]]
        :return: A matrix containing the minimum distances between each pair of boxes.
        :rtype: np.ndarray
        """
        # Precompute middle points for all boxes
        points1 = self._mid_points(class1_boxes)
        points2 = self._mid_points(class2_boxes)

        # Initialize an empty distance matrix
        num_class1 = len(class1_boxes)
        num_class2 = len(class2_boxes)
        distance_matrix = np.full((num_class1, num_class2), np.inf)

        # Calculate the minimum distance and fill the distance matrix
        for i in range(num_class1):
            for j in range(num_class2):
                distances = self._pair_distances(points1[i], points2[j])
                min_distance = np.min(distances)
                distance_matrix[i, j] = min_distance
        return distance_matrix

    def find_pairs(self, class1_boxes: Box, class2_boxes: Box) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the optimal pairing of boxes from from two classes that minimizes
        the total edge-to-edge distance using the Hungarian algorithm.

        :param class1_boxes: Bounding boxes of the first class in format (x0, y0, x1, y1).
        :type class1_boxes: list[tuple[float, float, float, float]]
        :param class2_boxes: Bounding boxes of the second class in format (x0, y0, x1, y1).
        :type class2_boxes: list[tuple[float, float, float, float]]
        :return: Indices of boxes (class1_ind, class2_ind) for class 1 and class 2 that form the optimal pairing.
        :rtype: (np.ndarray, np.ndarray)
        """
        logger.info('Find bounding box pairs.')
        distance_matrix = self._min_edge_distances(class1_boxes, class2_boxes)
        class1_ind, class2_ind = linear_sum_assignment(distance_matrix)
        return class1_ind, class2_ind


def map_annotations_to_checkboxes(document: Document) -> Document:
    """Map the annotations to the checkboxes."""

    bbox_pairing = BboxPairing()
    checkbox_detector = CheckboxDetector()

    # Loop through the pages of the document
    for page in document.pages():
        # get image and annotations
        image = page.get_image()
        annotations = page.view_annotations()  # not sure why page.annotations is returning an empty list.

        # simulate labels that should be paired to checkboxes
        labels_with_checkboxes = [
            'Geschlecht_Person1',
            'Geschlecht_Person2',
            'Statsangehörigkeit_Person1',
            'Statsangehörigkeit_Person2',
        ]
        for annotation in annotations:
            if annotation.label.name in labels_with_checkboxes:
                annotation.label.is_linked_to_checkbox = True

        # just evaluate for annotations with label.is_linked_to_checkbox == True
        annotations = [annotation for annotation in annotations if annotation.label.is_linked_to_checkbox]
        annotation_boxes = np.array(
            [
                (int(a.bbox().x0_image), int(a.bbox().y0_image), int(a.bbox().x1_image), int(a.bbox().y1_image))
                for a in annotations
            ]
        )

        checkboxes, checked, _ = checkbox_detector(image)

        # pair the checkboxes to the annotations
        ann_boxes_ind, checkbox_ind = bbox_pairing.find_pairs(annotation_boxes, checkboxes)

        # convert the checkboxes from image coordinates to document coordinates
        checkboxes = [
            (box.x0, box.x1, box.y0, box.y1)
            for box in [Bbox.from_image_size(x0, x1, y0, y1, page=page) for x0, y0, x1, y1 in checkboxes]
        ]

        # update the metadata of the annotations with the checkbox information
        for ann_idx, chbx_idx in zip(ann_boxes_ind, checkbox_ind):
            chbx_meta = {
                'omr': {
                    'is_checked': checked[chbx_idx],
                    'checkbox_bbox': checkboxes[chbx_idx],
                }
            }
            annotations[ann_idx].metadata = chbx_meta

    return document


if __name__ == '__main__':
    from PIL import Image, ImageDraw

    from konfuzio_sdk.data import Document, Project

    OMR_TEST_PROJECT_ID = 14848
    OMR_TEST_DOCUMENT_ID = 5772921

    project = Project(id_=OMR_TEST_PROJECT_ID)
    # project = Project(id_=OMR_TEST_PROJECT_ID, update=True)

    doc = project.get_document_by_id(OMR_TEST_DOCUMENT_ID)

    doc_with_anns = map_annotations_to_checkboxes(doc)

    for page in doc_with_anns.pages():
        image = page.get_image()
        annotations = page.view_annotations()

        image = image.convert('RGB')
        img_draw = ImageDraw.Draw(image)

        check_color = (0, 255, 0)
        uncheck_color = (0, 255, 255)
        label_color = (255, 0, 0)
        pair_colors = (255, 0, 255)

        scaler = max((image.size[0] / page.width, image.size[1] / page.height))

        for ann in annotations:
            # draw all annotation boxes
            an_x0, an_y0, an_x1, an_y1 = (
                ann.bbox().x0_image,
                ann.bbox().y0_image,
                ann.bbox().x1_image,
                ann.bbox().y1_image,
            )
            img_draw.rectangle((an_x0, an_y0, an_x1, an_y1), outline='blue', width=2)

            # draw all checkbox boxes and color code based on checked or not
            if ann.metadata:
                box_page = ann.metadata['omr']['checkbox_bbox']
                x0, y0, x1, y1 = box_page[0], box_page[2], box_page[1], box_page[3]
                x0, y0, x1, y1 = int(x0 * scaler), int(y0 * scaler), int(x1 * scaler), int(y1 * scaler)
                # coordinate system starts from bottom left, convert from top left to bottom left
                y0, y1 = image.size[1] - y1, image.size[1] - y0

                if ann.metadata['omr']['is_checked']:
                    img_draw.rectangle((x0, y0, x1, y1), outline=check_color, width=2)
                else:
                    img_draw.rectangle((x0, y0, x1, y1), outline=uncheck_color, width=2)

                # draw a point from the edge of the checkbox to the edge of the annotation
                # shortest distance from the edge of the checkbox to the edge of the annotation
                ed_x0 = max(x0, an_x0)
                ed_x1 = min(x1, an_x1)
                ed_y0 = max(y0, an_y0)
                ed_y1 = min(y1, an_y1)
                ed_y_ave = (ed_y0 + ed_y1) / 2
                img_draw.line((ed_x0, ed_y_ave, ed_x1, ed_y_ave), fill=pair_colors, width=2)

        image.save(str(f'./{page.document.name}_paired.png'))
