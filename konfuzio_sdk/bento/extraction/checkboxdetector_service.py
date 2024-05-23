"""Run checkbox detection service."""
import base64
import logging
from io import BytesIO

import bentoml
import debugpy
import numpy as np
from PIL import Image

from konfuzio_sdk.trainer.omr import BboxPairing

from .schemas import OMRRequest20240215, OMRResponse20240215

try:
    debugpy.listen(('0.0.0.0', 5678))  # Listen on the port defined in launch.json
    print('Waiting for debugger attach')
    debugpy.wait_for_client()
except Exception as e:
    print(f'Error while waiting for debugger attach: {e}\n This might be to a reload by bentoml serve.')


extraction_runner = bentoml.torchscript.get('checkboxdetector:latest').to_runner(embedded=True)
custom_objects = bentoml.torchscript.get('checkboxdetector:latest').custom_objects
detector_utils = custom_objects['utils']()
bbox_pairing = BboxPairing()


svc = bentoml.Service('extraction_svc', runners=[extraction_runner])

logger = logging.getLogger(__name__)


@svc.api(
    input=bentoml.io.JSON(pydantic_model=OMRRequest20240215),
    output=bentoml.io.JSON(pydantic_model=OMRResponse20240215),
)
async def extract(request: OMRRequest20240215) -> OMRResponse20240215:
    """Send an asynchronous call to the Extraction AI and process the response."""
    metadata = {}

    for page in request.pages:
        page_image = Image.open(BytesIO(base64.b64decode(page.image)))
        page_image = page_image.convert('RGB')

        annotations = [a for a in request.annotations if a.page_id == page.page_id]

        image_size = page_image.size
        page_size = (page.width, page.height)

        # convert the annotation bbox from page to image coordinates
        annotation_boxes = [
            (coords_page2img(a.bbox.x0, a.bbox.x1, a.bbox.y0, a.bbox.y1, page_size, image_size)) for a in annotations
        ]
        annotation_boxes = [(x0, y0, x1, y1) for x0, x1, y0, y1 in annotation_boxes]
        annotation_boxes = np.array(annotation_boxes)

        # bbox_pairing = BboxPairing()
        # checkbox_detector = CheckboxDetector()

        # result = extraction_runner.extract.async_run(page_image)
        image_tensor = detector_utils._preprocess(image=page_image, out_shape=(1280, 1280))
        outputs = extraction_runner.run(image_tensor)
        cls_conf, checkboxes = detector_utils._postprocess(outputs, page_image.size)
        checked = [True if c[0] > c[1] else False for c in cls_conf]
        confidence = [max(c) for c in cls_conf]
        # pair the checkboxes to the annotations
        ann_boxes_ind, checkbox_ind = bbox_pairing.find_pairs(annotation_boxes, checkboxes)

        # convert the checkboxes from image coordinates to document coordinates
        checkboxes = [coords_img2page(x0, x1, y0, y1, page_size, image_size) for x0, y0, x1, y1 in checkboxes]

        # update the metadata of the annotations with the checkbox information
        for ann_idx, chbx_idx in zip(ann_boxes_ind, checkbox_ind):
            chbx_meta = {
                'omr': {
                    'is_checked': checked[chbx_idx],
                    'checkbox_bbox': checkboxes[chbx_idx],
                    'confidence': confidence[chbx_idx],
                }
            }

            a_id_ = annotations[ann_idx].annotation_id
            metadata[a_id_] = chbx_meta

    return metadata


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
