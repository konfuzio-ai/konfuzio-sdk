"""Run checkbox detection service."""

import base64
import logging
import os
from io import BytesIO
from typing import Any

import bentoml
import numpy as np
from fastapi import FastAPI
from PIL import Image
from trainer.omr import CheckboxDetectorUtils  # import from the built bento directory src/trainer/omr.py

from konfuzio_sdk.trainer.omr import BboxPairing

from .schemas import CheckboxRequest20240523, CheckboxResponse20240523

# load ai model name AI_MODEL_NAME file in parent directory
ai_model_name_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'AI_MODEL_NAME')
ai_model_name = open(ai_model_name_file).read().strip()

app = FastAPI()
logger = logging.getLogger(__name__)


@bentoml.service
@bentoml.mount_asgi_app(app, path='/v1')
class CheckboxService:
    def __init__(self) -> None:
        """Load the checkbox model into memory."""
        self.extraction_model = bentoml.torchscript.load_model(ai_model_name + ':latest')
        self.detector_utils = CheckboxDetectorUtils()
        self.bbox_pairing = BboxPairing()

    @bentoml.api(input_spec=CheckboxRequest20240523)
    async def extract(self, **request: Any) -> CheckboxResponse20240523:
        """Send an call to the CheckboxDetector and process the response."""

        request = CheckboxRequest20240523(**request)

        metadata = []

        for page in request.pages:
            page_image = Image.open(BytesIO(base64.b64decode(page.image)))
            page_image = page_image.convert('RGB')

            annotations = [a for a in request.annotations if a.page_id == page.page_id]

            image_size = page_image.size
            page_size = (page.width, page.height)

            # convert the annotation bbox from page to image coordinates
            annotation_boxes = [
                (coords_page2img(a.bbox.x0, a.bbox.x1, a.bbox.y0, a.bbox.y1, page_size, image_size))
                for a in annotations
            ]
            annotation_boxes = [(x0, y0, x1, y1) for x0, x1, y0, y1 in annotation_boxes]
            annotation_boxes = np.array(annotation_boxes)

            image_tensor = self.detector_utils._preprocess(image=page_image, out_shape=(1280, 1280))
            outputs = self.extraction_model(image_tensor)
            cls_conf, checkboxes = self.detector_utils._postprocess(outputs, page_image.size)
            checked = [True if c[0] > c[1] else False for c in cls_conf]
            confidence = [max(c) for c in cls_conf]
            # pair the checkboxes to the annotations
            ann_boxes_ind, checkbox_ind = self.bbox_pairing.find_pairs(annotation_boxes, checkboxes)

            # convert the checkboxes from image coordinates to document coordinates
            checkboxes = [coords_img2page(x0, x1, y0, y1, page_size, image_size) for x0, y0, x1, y1 in checkboxes]
            checkboxes = [{'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1} for x0, x1, y0, y1 in checkboxes]

            # update the metadata of the annotations with the checkbox information
            for ann_idx, chbx_idx in zip(ann_boxes_ind, checkbox_ind):
                chbx_meta = {
                    'is_checked': checked[chbx_idx],
                    'bbox': checkboxes[chbx_idx],
                    'confidence': float(confidence[chbx_idx]),
                }
                a_id_ = annotations[ann_idx].annotation_id
                metadata.append({'annotation_id': a_id_, 'checkbox': chbx_meta})

        return CheckboxResponse20240523(metadata=metadata)


def coords_img2page(x0, x1, y0, y1, page_shape, image_shape):
    """Convert and scale the coordinates from image to page."""
    (page_w, page_h) = page_shape
    (image_w, image_h) = image_shape

    scale_y = page_h / image_h
    scale_x = page_w / image_w

    # scale
    y0, y1 = y0 * scale_y, y1 * scale_y
    x0, x1 = x0 * scale_x, x1 * scale_x
    # convert
    y0, y1 = page_h - y1, page_h - y0

    return int(x0), int(x1), int(y0), int(y1)


def coords_page2img(x0, x1, y0, y1, page_shape, image_shape):
    """Convert and scale the coordinates from page to image coordinates."""
    (page_w, page_h) = page_shape
    (image_w, image_h) = image_shape

    scale_y = image_h / page_h
    scale_x = image_w / page_w
    # scale
    y0, y1 = y0 * scale_y, y1 * scale_y
    x0, x1 = x0 * scale_x, x1 * scale_x
    # convert
    y0, y1 = image_h - y1, image_h - y0

    return int(x0), int(x1), int(y0), int(y1)
