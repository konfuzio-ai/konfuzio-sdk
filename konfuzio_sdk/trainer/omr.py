"""Optical Mark Recognition (OMR) for processing checkboxes."""

import abc
import logging
import pathlib
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.optimize import linear_sum_assignment

from konfuzio_sdk.data import Bbox, Document
from konfuzio_sdk.extras import FloatTensor, Module

logger = logging.getLogger(__name__)
logger.info('Creating phrase matcher')
logger.error('[ERROR] Creating phrase matcher')


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

    @abc.abstractmethod
    def forward(self, image: Image.Image) -> Any:
        """
        Abstract method to be implemented by subclasses. Defines the forward pass of the model.

        :param image: The input image to process.
        :type image: PIL.Image.Image
        :return: The result of the model prediction.
        :rtype: Any
        """


class CheckboxDetector(OMRAbstractModel):
    """Detects checkboxes in images using a pre-trained model."""

    def __init__(
        self,
        **kwargs,
    ):
        """Initializes the CheckboxDetector with a pre-trained model and default parameters."""
        super().__init__()
        path = str(pathlib.Path(__file__).parent / 'ckpt_best.pt')
        self.detector = torch.jit.load(path)
        self.input_shape = (1280, 1280)
        self.threshold = 0.7

    def _threshold(self, cls_conf, bboxes):
        idx = np.argwhere(cls_conf > self.threshold)
        cls_conf = cls_conf[idx[:, 0]]
        bboxes = bboxes[idx[:, 0]]
        return cls_conf, bboxes

    def _nms(self, cls_conf, bboxes):
        indices = torchvision.ops.nms(
            torch.from_numpy(bboxes), torch.from_numpy(cls_conf.max(1)), iou_threshold=0.5
        ).numpy()
        cls_conf = cls_conf[indices]
        bboxes = bboxes[indices]
        return cls_conf, bboxes

    def _preprocess(self, image: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
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

    def _postprocess(self, outputs, image_shape: Tuple[int, int]):
        cls_conf = outputs[1][0].cpu().detach().numpy()
        bboxes = outputs[0][0].cpu().detach().numpy()

        cls_conf, bboxes = self._threshold(cls_conf, bboxes)

        cls_conf, bboxes = self._nms(cls_conf, bboxes)

        # Define and apply scale for the bounding boxes to the original image size
        scaler = max((image_shape[1] / self.input_shape[1], image_shape[0] / self.input_shape[0]))
        bboxes *= scaler
        bboxes = np.array([(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in bboxes])
        return cls_conf, bboxes

    def _rescale(self, image: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
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
        input_shape = image.shape[:2]

        pad_height = output_shape[0] - input_shape[0]
        pad_width = output_shape[1] - input_shape[1]

        pad_h = (0, pad_height)  # top=0, bottom=pad_height
        pad_w = (0, pad_width)  # left=0, right=pad_width

        _, _, num_channels = image.shape

        constant_values = ((pad_value, pad_value), (pad_value, pad_value), (0, 0))
        # Fixes issue with numpy deprecation warning since constant_values is ragged array (Have to explicitly specify object dtype)
        constant_values = np.array(constant_values, dtype=np.object_)

        padding_values = (pad_h, pad_w, (0, 0))

        processed_image = np.pad(image, pad_width=padding_values, mode='constant', constant_values=constant_values)

        return processed_image

    def _permute(self, image: np.ndarray, permutation: Tuple[int, int, int] = (2, 0, 1)) -> np.ndarray:
        processed_image = np.ascontiguousarray(image.transpose(permutation))
        return processed_image

    def _standardize(self, image: np.ndarray, max_value=255) -> np.ndarray:
        processed_image = (image / max_value).astype(np.float32)
        return processed_image

    def forward(self, image: Image.Image) -> Dict[str, FloatTensor]:
        input = self._preprocess(image, self.input_shape)
        outputs = self.detector(input)
        cls_conf, bboxes = self._postprocess(outputs, image.size)
        checked = [True if c[0] > c[1] else False for c in cls_conf]

        return bboxes, checked, cls_conf


class BboxPairing:
    """Handles bounding box pairing in the format of (x1, y1, x2, y2)."""

    def __init__(self):
        pass

    def _mid_points(self, boxes):
        """
        For each box, calculate the middle points of its edges.
        Each box is defined as (x1, y1, x2, y2).
        Returns a list of arrays, each containing four points (top, bottom, left, right) for each box.
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

    def _pair_distances(self, middle_points1, middle_points2):
        """
        Calculate distances between all pairs of middle points from two sets of boxes.
        `middle_points1` and `middle_points2` are arrays of middle points precomputed.
        """
        # Expand dimensions to enable broadcasting: (n_points1, 1, 2) and (1, n_points2, 2)
        expanded_points1 = np.expand_dims(middle_points1, 1)
        expanded_points2 = np.expand_dims(middle_points2, 0)

        # Compute distances using broadcasting
        distances = np.sqrt(np.sum((expanded_points1 - expanded_points2) ** 2, axis=2))
        return distances

    def _min_edge_distances(self, class1_boxes, class2_boxes):
        """
        Calculate the minimum edge-to-edge distance between boxes in class 1 and class 2.
        """
        # Precompute middle points for all boxes
        points1 = self._mid_points(class1_boxes)
        points2 = self._mid_points(class2_boxes)

        # Initialize an empty distance matrix
        num_class1 = len(class1_boxes)
        num_class2 = len(class2_boxes)

        distance_matrix = np.full((num_class1, num_class2), np.inf)

        # For each box in class 1, calculate distance to each box in class 2
        for i in range(num_class1):
            for j in range(num_class2):
                # Calculate pairwise distances between edges of the two boxes
                distances = self._pair_distances(points1[i], points2[j])
                # Find the minimum distance for this box pair
                min_distance = np.min(distances)
                distance_matrix[i, j] = min_distance
        return distance_matrix

    def find_pairs(self, class1_boxes, class2_boxes):
        """
        Find the optimal pairing of boxes from class 2 to class 1 that minimizes
        the total edge-to-edge distance using the Hungarian algorithm.
        """
        distance_matrix = self._min_edge_distances(class1_boxes, class2_boxes)
        class1_ind, class2_ind = linear_sum_assignment(distance_matrix)
        return class1_ind, class2_ind


def map_annotations_to_checkboxes(document: Document) -> Document:
    """Map the annotations to the checkboxes.

    Args:
        document: Document object containing the annotations and the page image.

    Returns:
        Document object containing the annotations and the page image with the mapped checkboxes.
    """

    bbox_pairing = BboxPairing()
    checkbox_detector = CheckboxDetector()

    # Loop through the pages of the document
    for page in document.pages():
        # get image and annotations
        image = page.get_image()
        annotations = page.view_annotations()  # not sure why page.annotations is returning an empty list.

        # simulate labels that are linked to checkboxes
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
        annotation_boxes = {
            a.id_: (int(a.bbox().x0_image), int(a.bbox().y0_image), int(a.bbox().x1_image), int(a.bbox().y1_image))
            for a in annotations
        }

        checkboxes, checked, _ = checkbox_detector(image)

        # link the checkboxes to the annotations
        annotation_boxes_list = list(annotation_boxes.values())
        checkbox_list = list(checkboxes)
        ann_boxes_ind, checkbox_ind = bbox_pairing.find_pairs(annotation_boxes_list, checkbox_list)

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
        link_colors = (255, 0, 255)

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
                img_draw.line((ed_x0, ed_y_ave, ed_x1, ed_y_ave), fill=link_colors, width=2)

        image.save(str(f'./{page.document.name}_linked.png'))
