"""Utils for handling bounding boxes."""

import itertools
import logging

logger = logging.getLogger(__name__)


def convert_segmentation_bbox(bbox: dict, page: dict) -> dict:
    """
    Convert bounding box from the segmentation result to the scale of the characters bboxes of the document.

    :param bbox: Bounding box from the segmentation result
    :param page: Page information
    :return: Converted bounding box.
    """
    original_size = page['original_size']
    image_size = page['size']
    factor_y = original_size[1] / image_size[1]
    factor_x = original_size[0] / image_size[0]
    height = image_size[1]

    temp_y0 = (height - bbox['y0']) * factor_y
    temp_y1 = (height - bbox['y1']) * factor_y
    bbox['y0'] = temp_y1
    bbox['y1'] = temp_y0
    bbox['x0'] = bbox['x0'] * factor_x
    bbox['x1'] = bbox['x1'] * factor_x

    return bbox


def select_bboxes(selection_bbox: dict, page_bboxes: list, tolerance: int = 10) -> list:
    """
    Filter the characters bboxes of the document page according to their x/y values.

    The result only includes the characters that are inside the selection bbox.

    :param selection_bbox: Bounding box used to select the characters bboxes.
    :param page_bboxes: Bounding boxes of the characters in the document page.
    :param tolerance: Tolerance for the coordinates values.
    :return: Selected characters bboxes.
    """
    selected_char_bboxes = [
        char_bbox
        for char_bbox in page_bboxes
        if int(selection_bbox["x0"]) - tolerance <= char_bbox["x0"]
        and int(selection_bbox["x1"]) + tolerance >= char_bbox["x1"]
        and int(selection_bbox["y0"]) - tolerance <= char_bbox["y0"]
        and int(selection_bbox["y1"]) + tolerance >= char_bbox["y1"]
    ]

    return selected_char_bboxes


def group_bboxes_per_line(char_bboxes: dict, page_index: int) -> list:
    """
    Group characters bounding boxes per line.

    A line will have a single bounding box.

    :param char_bboxes: Bounding boxes of the characters.
    :param page_index: Index of the page in the document.
    :return: List with 1 bounding box per line.
    """
    lines_bboxes = []

    # iterate over each line_number and all of the character bboxes with that line number
    for line_number, line_char_bboxes in itertools.groupby(char_bboxes, lambda x: x['line_number']):
        # set the defaut values which we overwrite with the actual character bbox values
        x0 = 100000000
        top = 10000000
        y0 = 10000000
        x1 = 0
        y1 = 0
        bottom = 0
        start_offset = 100000000
        end_offset = 0

        # remove space chars from the line selection so they don't interfere with the merging of bboxes
        # (a bbox should never start with a space char)
        trimmed_line_char_bboxes = [char for char in line_char_bboxes if not char['text'].isspace()]

        if len(trimmed_line_char_bboxes) == 0:
            continue

        # merge characters bounding boxes of the same line
        for char_bbox in trimmed_line_char_bboxes:
            x0 = min(char_bbox['x0'], x0)
            top = min(char_bbox['top'], top)
            y0 = min(char_bbox['y0'], y0)

            x1 = max(char_bbox['x1'], x1)
            bottom = max(char_bbox['bottom'], bottom)
            y1 = max(char_bbox['y1'], y1)

            start_offset = min(int(char_bbox['string_offset']), start_offset)
            end_offset = max(int(char_bbox['string_offset']), end_offset)

        line_bbox = {
            'bottom': bottom,
            'page_index': page_index,
            'top': top,
            'x0': x0,
            'x1': x1,
            'y0': y0,
            'y1': y1,
            'start_offset': start_offset,
            'end_offset': end_offset + 1,
            'line_number': line_number,
        }

        lines_bboxes.append(line_bbox)

    return lines_bboxes


def merge_bboxes(bboxes: list):
    """
    Merge bounding boxes.

    :param bboxes: Bounding boxes to be merged.
    :return: Merged bounding box.
    """
    merge_bbox = {
        "x0": min([b['x0'] for b in bboxes]),
        "x1": max([b['x1'] for b in bboxes]),
        "y0": min([b['y0'] for b in bboxes]),
        "y1": max([b['y1'] for b in bboxes]),
        "top": min([b['top'] for b in bboxes]),
        "bottom": max([b['bottom'] for b in bboxes]),
        "page_index": bboxes[0]['page_index'],
    }

    return merge_bbox
