"""Handle data from the API."""

import io
import itertools
import json
import logging
import os
import pathlib
import shutil
import time
import zipfile
from copy import deepcopy
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple, Union
from warnings import warn

import dateutil.parser
import regex as re
from PIL import Image, ImageDraw, ImageFont
from requests import HTTPError
from tqdm import tqdm

from konfuzio_sdk.api import (
    delete_document_annotation,
    delete_file_konfuzio_api,
    download_file_konfuzio_api,
    export_ai_models,
    get_all_project_ais,
    get_document_annotations,
    get_document_bbox,
    get_document_details,
    get_meta_of_files,
    get_page_image,
    get_project_categories,
    get_results_from_segmentation,
    konfuzio_session,
    post_document_annotation,
    update_document_konfuzio_api,
    upload_file_konfuzio_api,
)
from konfuzio_sdk.normalize import normalize
from konfuzio_sdk.regex import get_best_regex, merge_regex, regex_matches, suggest_regex_for_string
from konfuzio_sdk.urls import get_annotation_view_url
from konfuzio_sdk.utils import (
    amend_file_name,
    convert_to_bio_scheme,
    exception_or_log_error,
    get_file_type_and_extension,
    get_merged_bboxes,
    get_missing_offsets,
    is_file,
    sdk_isinstance,
)

logger = logging.getLogger(__name__)


class Data:
    """Collect general functionality to work with data from API."""

    id_iter = itertools.count()
    id_ = None
    id_local = None
    session = konfuzio_session()
    _update = False
    _force_offline = False

    def __eq__(self, other) -> bool:
        """Compare any point of Data with their ID, overwrite if needed."""
        if self.id_ is None and other and other.id_ is None:
            # Compare to virtual instances
            return self.id_local == other.id_local
        else:
            return self.id_ is not None and other is not None and other.id_ is not None and self.id_ == other.id_

    def __hash__(self):
        """Make any online or local concept hashable. See https://stackoverflow.com/a/7152650."""
        return hash(str(self.id_local))

    def __copy__(self):
        """Not yet modelled."""
        raise NotImplementedError

    def __deepcopy__(self, memodict):
        """Not yet modelled."""
        raise NotImplementedError

    @property
    def is_online(self) -> Optional[int]:
        """Define if the Document is saved to the server."""
        return (self.id_ is not None) and (not self._force_offline)

    # todo require to overwrite lose_weight via @abstractmethod
    def lose_weight(self):
        """Delete data of the instance."""
        self.session = None
        return self

    def set_offline(self):
        """Force Data into offline mode."""
        self._force_offline = True
        self._update = False


class Page(Data):
    """
    Access the information about one Page of a Document.

    For more details see https://dev.konfuzio.com/sdk/explanations.html#page-concept
    """

    def __init__(
        self,
        id_: Union[int, None],
        document: 'Document',
        number: int,
        original_size: Tuple[float, float],
        image_size: Tuple[int, int] = (None, None),
        start_offset: Optional[int] = None,
        end_offset: Optional[int] = None,
        copy_of_id: Optional[int] = None,
    ):
        """
        Create a Page for a Document.

        :param id_: ID of the Page
        :param document: Document the Page belongs to
        :param number: Page number in Document (1-based indexing)
        :param original_size: Size of original Document file Page (all Document Bboxes are based on this scale)
        :param image_size: Size of the image representation of the Page
        :param start_offset: Start of the Page in the text of the Document
        :param end_offset: End of the Page in the text of the Document
        :param category: The Category the Page belongs to, if any
        :param copy_of_id: The ID of the Page that this Page is a copy of, if any
        """
        self.id_ = id_
        self.document = document
        self.number = number
        self.index = number - 1
        document.add_page(self)
        self.start_offset = start_offset
        self.end_offset = end_offset
        if start_offset is None or end_offset is None:
            page_texts = self.document.text.split('\f')
            self.start_offset = sum([len(page_text) for page_text in page_texts[: self.index]]) + self.index
            self.end_offset = self.start_offset + len(page_texts[self.index])

        self.copy_of_id = copy_of_id
        self.text_encoded: List[int] = None
        self.image: Optional[Image.Image] = None
        self.image_bytes: Optional[bytes] = None
        self._original_size = original_size
        self.width = self._original_size[0]
        self.height = self._original_size[1]
        self._image_size = image_size
        self.image_width = self._image_size[0]
        self.image_height = self._image_size[1]

        document_folder = (
            self.document.document_folder
            if self.id_
            else self.document.project.get_document_by_id(self.document.copy_of_id).document_folder
        )
        self.image_path = os.path.join(document_folder, f'page_{self.number}.png')

        self._category = self.document._category
        self.category_annotations: List['CategoryAnnotation'] = []
        self._human_chosen_category_annotation: Optional[CategoryAnnotation] = None
        self._segmentation = None
        self.is_first_page = None
        self.is_first_page_confidence = None
        if self.number == 1:
            self.is_first_page = True
            self.is_first_page_confidence = 1
        else:
            self.is_first_page = False

        check_page = True
        if self.index is None:
            logger.error(f'Page index is None of {self} in {self.document}.')
            check_page = False
        if self.height is None:  # todo why do we allow pages with height<=0?
            logger.error(f'Page Height is None of {self} in {self.document}.')
            check_page = False
        if self.width is None:  # todo why do we allow pages with width<=0?
            logger.error(f'Page Width is None of {self} in {self.document}.')
            check_page = False
        assert check_page

    def __hash__(self):
        """Define that one Page per Document is unique."""
        return hash((self.document, self.index))

    def __eq__(self, other: 'Page') -> bool:
        """Define how one Page is identical."""
        return self.__hash__() == other.__hash__()

    def __repr__(self):
        """Return the name of the Document incl. the ID."""
        return f'Page {self.index} in {self.document}'

    def get_image(self, update: bool = False) -> Image.Image:
        """
        Get Page as a Pillow Image object.

        The Page image is loaded from a PNG file at `Page.image_path`.
        If the file is not present, or if `update` is True, it will be downloaded from the Konfuzio Host.
        Alternatively, if you don't want to use a file, you can provide the image as bytes to `Page.image_bytes`. Then
        call this method to convert the bytes into a Pillow Image.
        In every case, the return value of this method and the attribute `Page.image` will be a Pillow Image.

        :param update: Whether to force download the Page PNG file.
        :return: A Pillow Image object for this Page's image.
        """
        if not self.image or update:
            page_id = self.id_ if self.id_ else self.copy_of_id
            if self.image_bytes:
                try:
                    self.image = Image.open(io.BytesIO(self.image_bytes))
                except ValueError:
                    self.image = Image.open(io.BytesIO(self.image_bytes)).convert('RGB')
            elif is_file(self.image_path, raise_exception=False) and not update:
                self.image = Image.open(self.image_path)
            elif (not is_file(self.image_path, raise_exception=False) or update) and page_id:
                document_id = self.document.id_ if self.document.id_ else self.document.copy_of_id
                png_content = get_page_image(
                    document_id=document_id, page_number=self.number, session=self.document.project.session
                )
                with open(self.image_path, 'wb') as f:
                    f.write(png_content)
                    self.image = Image.open(io.BytesIO(png_content))

        return self.image

    def get_annotations_image(self, display_all: bool = False) -> Image:
        """Get Document Page as PNG with Annotations shown."""
        image = self.get_image()
        image = image.convert('RGB')

        draw = ImageDraw.Draw(image)

        try:
            # We try to get a ttf font to be able to change bounding box label text size
            font = ImageFont.truetype(
                '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 24, encoding='unic'
            )
        except OSError:
            logger.warning('Font not found. Loading default.')
            font = ImageFont.load_default()

        if not display_all:
            annotations = self.view_annotations()
        else:
            annotations = self.annotations(use_correct=False)
        for annotation in annotations:
            annotation_image_bbox = (
                annotation.bbox().x0_image,
                annotation.bbox().y0_image,
                annotation.bbox().x1_image,
                annotation.bbox().y1_image,
            )
            draw.rectangle(annotation_image_bbox, outline='blue', width=2)
            draw.text(
                (annotation_image_bbox[0], annotation_image_bbox[1] - 24), annotation.label.name, fill='blue', font=font
            )
            for span in annotation.spans:
                span_image_bbox = (
                    span.bbox().x0_image,
                    span.bbox().y0_image,
                    span.bbox().x1_image,
                    span.bbox().y1_image,
                )
                draw.rectangle(span_image_bbox, outline='green', width=1)

        return image

    @property
    def text(self):
        """Get Document text corresponding to the Page."""
        doc_text = self.document.text
        page_text = self.document.text[self.start_offset : self.end_offset]
        if doc_text.split('\f')[self.index] != page_text:
            raise IndexError(f'{self} text offsets do not match Document text.')
        return page_text

    @property
    def number_of_lines(self) -> int:
        """Calculate the number of lines in Page."""
        return len(self.text.split('\n'))

    def spans(
        self,
        label: 'Label' = None,
        use_correct: bool = False,
        start_offset: int = 0,
        end_offset: int = None,
        fill: bool = False,
    ) -> List['Span']:
        """Return all Spans of the Page."""
        spans = []
        annotations = self.annotations(
            label=label, use_correct=use_correct, start_offset=start_offset, end_offset=end_offset, fill=fill
        )
        for annotation in annotations:
            for span in annotation.spans:
                if span not in spans:
                    spans.append(span)

        return sorted(spans)

    def lines(self) -> List['Span']:
        """Return sorted list of Spans for each line in the Page."""
        lines_spans = []
        char_bboxes = self.get_bbox().values()
        char_bboxes = sorted(char_bboxes, key=lambda x: x['char_index'])

        # iterate over each line_number and all the character bboxes with that line number

        # condition for backward compatibility that checks if bbox file structure includes 'line_number' or
        # 'line_index' and parses the file accordingly.
        if 'line_number' in [bbox.keys() for bbox in char_bboxes][0]:
            for _, line_char_bboxes in itertools.groupby(char_bboxes, lambda x: x['line_number']):
                # (a line should never start with a space char)
                trimmed_line_char_bboxes = [char for char in line_char_bboxes if not char['text'].isspace()]

                if len(trimmed_line_char_bboxes) == 0:
                    continue

                # create Span from the line characters bboxes
                start_offset = min((char_bbox['char_index'] for char_bbox in trimmed_line_char_bboxes))
                end_offset = max((char_bbox['char_index'] for char_bbox in trimmed_line_char_bboxes)) + 1
                span = Span(start_offset=start_offset, end_offset=end_offset, document=self.document)

                lines_spans.append(span)
        elif 'line_index' in [bbox.keys() for bbox in char_bboxes][0]:
            # line index is always less than line_index by 1, e.g. if line_index is 0, line_number is 1
            for _, line_char_bboxes in itertools.groupby(char_bboxes, lambda x: x['line_index'] + 1):
                # (a line should never start with a space char)
                trimmed_line_char_bboxes = [char for char in line_char_bboxes if not char['text'].isspace()]

                if len(trimmed_line_char_bboxes) == 0:
                    continue

                # create Span from the line characters bboxes
                start_offset = min((char_bbox['char_index'] for char_bbox in trimmed_line_char_bboxes))
                end_offset = max((char_bbox['char_index'] for char_bbox in trimmed_line_char_bboxes)) + 1
                span = Span(start_offset=start_offset, end_offset=end_offset, document=self.document)

                lines_spans.append(span)

        return lines_spans

    def get_bbox(self):
        """Get bbox information per character of Page."""
        page_bbox = self.document.get_bbox_by_page(self.index)
        return page_bbox

    def annotations(
        self,
        label: 'Label' = None,
        use_correct: bool = True,
        ignore_below_threshold: bool = False,
        start_offset: int = 0,
        end_offset: int = None,
        fill: bool = False,
    ) -> List['Annotation']:
        """Get Page Annotations."""
        start_offset = max(start_offset, self.start_offset)
        if end_offset is None:
            end_offset = self.end_offset
        else:
            end_offset = min(end_offset, self.end_offset)
        page_annotations = self.document.annotations(
            label=label,
            use_correct=use_correct,
            ignore_below_threshold=ignore_below_threshold,
            start_offset=start_offset,
            end_offset=end_offset,
            fill=fill,
        )
        return page_annotations

    def view_annotations(self) -> List['Annotation']:
        """Get the best Annotations, where the Spans are not overlapping in Page."""
        page_view_anns = self.document.view_annotations(start_offset=self.start_offset, end_offset=self.end_offset)
        return page_view_anns

    def add_category_annotation(self, category_annotation: 'CategoryAnnotation'):
        """Annotate a Page with a Category and confidence information."""
        if category_annotation.category != self.document.project.no_category:
            duplicated = [x for x in self.category_annotations if x == category_annotation]
            if duplicated:
                raise ValueError(
                    f'In {self} the {category_annotation} is a duplicate of {duplicated} and will not be added.'
                )
            self.category_annotations.append(category_annotation)

    def get_category_annotation(self, category, add_if_not_present: bool = False) -> 'CategoryAnnotation':
        """
        Retrieve the Category Annotation associated with a specific Category within this Page.

        If no Category Annotation is found for the provided Category, one can be created based on the
        `add_if_not_present` argument.

        :param category: The Category for which to retrieve the Category Annotation.
        :type category: Category
        :param add_if_not_present: If True, a Category Annotation will be added to the current Page if none is found. If
                                False, a dummy Category Annotation will be created, not linked to any Document or Page.
        :type add_if_not_present: bool

        :return: The located or newly created Category Annotation.
        :rtype: CategoryAnnotation
        """
        filtered_category_annotations = [
            category_annotation
            for category_annotation in self.category_annotations
            if category_annotation.category == category
            and category_annotation.category != self.document.project.no_category
        ]
        # if the list is not empty it means there is exactly one CategoryAnnotation with the assigned Category
        # (see Page.add_category_annotation for duplicate checking)
        if filtered_category_annotations:
            return filtered_category_annotations[0]
        else:  # otherwise a new one will be created
            if add_if_not_present:
                new_category_annotation = CategoryAnnotation(category=category, page=self)
            else:
                # dummy CategoryAnnotation (not associated to any Document or Page)
                new_category_annotation = CategoryAnnotation(category=category)
            return new_category_annotation

    def set_category(self, category: 'Category') -> None:
        """
        Set the Category of the Page.

        :param category: The Category to set for the Page.
        """
        if not category:
            raise ValueError("We forbid setting a Page's Category to None.")
        logger.info(f'Setting {self} Category to {category}.')
        self._category = category
        if category is self.document.project.no_category:
            self.category_annotations = []
            self._human_chosen_category_annotation = None
            return
        category_annotation = self.get_category_annotation(category, add_if_not_present=True)
        self._human_chosen_category_annotation = category_annotation

    @property
    def maximum_confidence_category_annotation(self) -> Optional['CategoryAnnotation']:
        """
        Get the human revised Category Annotation of this Page, or the highest confidence one if not revised.

        :return: The found Category Annotation, or None if not present.
        """
        if (
            self._human_chosen_category_annotation is not None
            and self._human_chosen_category_annotation.category != self.document.project.no_category
        ):
            return self._human_chosen_category_annotation
        elif self.category_annotations:
            # return the highest confidence CategoryAnnotation if no human revised it
            return sorted(self.category_annotations, key=lambda x: x.confidence)[-1]
        else:
            return None

    @property
    def category(self) -> Optional['Category']:
        """Get the Category of the Page, based on human revised Category Annotation, or on highest confidence."""
        if self.maximum_confidence_category_annotation is not None:
            return self.maximum_confidence_category_annotation.category
        else:
            return self._category

    def get_original_page(self) -> 'Page':
        """
        Return an "original" Page in case the current Page is a copy without an ID.

        An "original" Page is a Page from the Document that is not a copy and not a Virtual Document. This Page has an
        ID.

        The method is used in the File Splitting pipeline to retain the original Document's information in
        the Sub-Documents that were created from its splitting. The original Document is a Document that has an ID and
        is not a deepcopy.
        """
        if self.id_ and self.document.id_:
            return self
        elif self.copy_of_id:
            if self.document.id_:
                return self.document.get_page_by_id(self.copy_of_id)
            else:
                return self.document.project.get_document_by_id(self.document.copy_of_id).get_page_by_id(
                    self.copy_of_id
                )

    def annotation_sets(self) -> List['AnnotationSet']:
        """Show all Annotation Sets related to Annotations of the Page."""
        page_annotations = self.annotations()
        return list({annotation.annotation_set for annotation in page_annotations})


class BboxValidationTypes(Enum):
    """
    Define validation strictness for Bounding Boxes.

    For more details see the `Bbox` class.
    """

    STRICT = 'strict'
    ALLOW_ZERO_SIZE = 'allow zero size'
    DISABLED = 'disabled'


class Bbox:
    """
    A bounding box relates to an area of a Document Page.

    For more details see https://dev.konfuzio.com/sdk/explanations.html#bbox-concept

    What consistutes a valid Bbox changes depending on the value of the `validation` param.
    If ALLOW_ZERO_SIZE (default), it allows bounding boxes to have zero width or height.
    This option is available for compatibility reasons since some OCR engines can sometimes return character level
    bboxes with zero width or height. If STRICT, it doesn't allow zero size bboxes. If DISABLED, it allows bboxes that
    have negative size, or coordinates beyond the Page bounds.
    For the default behaviour see https://dev.konfuzio.com/sdk/tutorials/data_validation/index.html

    :param validation: One of ALLOW_ZERO_SIZE (default), STRICT, or DISABLED.
    """

    def __init__(self, x0: int, x1: int, y0: int, y1: int, page: Page, validation=BboxValidationTypes.ALLOW_ZERO_SIZE):
        """Store information and validate."""
        self.x0: int = x0
        self.x1: int = x1
        self.y0: int = y0
        self.y1: int = y1
        self.angle: float = 0.0  # not yet used
        self.page: Page = page
        self._label_name: Optional[str] = None  # used in detectron tokenizer
        self._valid(validation)

    @property
    def document(self) -> 'Document':
        """Get the Document the Bbox belongs to."""
        return self.page.document

    @property
    def top(self):
        """Calculate the distance to the top of the Page."""
        if self.page:
            return round(self.page.height - self.y1, 3)

    def __repr__(self):
        """Represent the Box."""
        return f'{self.__class__.__name__}: x0: {self.x0} x1: {self.x1} y0: {self.y0} y1: {self.y1} on Page {self.page}'

    def __hash__(self):
        """Return identical value for a Bounding Box."""
        return hash((self.x0, self.x1, self.y0, self.y1, self.page))

    def __eq__(self, other: 'Bbox') -> bool:
        """Define that one Bounding Box on the same page is identical."""
        return self.__hash__() == other.__hash__()

    def _valid(self, validation=BboxValidationTypes.ALLOW_ZERO_SIZE, handler='sdk_validation'):
        """
        Validate the coordinates of the Bounding Box contained in the Bbox, raising a ValueError exception in case.

        :param validation: One of ALLOW_ZERO_SIZE (default), STRICT, or DISABLED. Also see the `Bbox` class.
        """
        round_decimals = 2

        if round(self.x0, round_decimals) == round(self.x1, round_decimals):
            logger.warning(f'{self} has no width in {self.page}.')

        if round(self.x0, round_decimals) > round(self.x1, round_decimals):
            exception_or_log_error(
                msg=f'{self} has negative width in {self.page}.',
                fail_loudly=str(validation) != str(BboxValidationTypes.DISABLED),
                exception_type=ValueError,
                handler=handler,
            )

        if round(self.y0, round_decimals) == round(self.y1, round_decimals):
            logger.warning(f'{self} has no height in {self.page}.')

        if round(self.y0, round_decimals) > round(self.y1, round_decimals):
            exception_or_log_error(
                msg=f'{self} has negative height in {self.page}.',
                fail_loudly=str(validation) != str(BboxValidationTypes.DISABLED),
                exception_type=ValueError,
                handler=handler,
            )

        if round(self.y1, round_decimals) > round(self.page.height, round_decimals):
            exception_or_log_error(
                msg=f'{self} exceeds height of {self.page} by '
                f'{round(self.y1, round_decimals) - round(self.page.height, round_decimals)} '
                f'with validation {validation}.',
                fail_loudly=str(validation) != str(BboxValidationTypes.DISABLED),
                exception_type=ValueError,
                handler=handler,
            )

        if round(self.x1, round_decimals) > round(self.page.width, round_decimals):
            exception_or_log_error(
                msg=f'{self} exceeds width of {self.page} by '
                f'{round(self.x1, round_decimals) - round(self.page.width, round_decimals)} '
                f'with validation {validation}.',
                fail_loudly=str(validation) != str(BboxValidationTypes.DISABLED),
                exception_type=ValueError,
                handler=handler,
            )

        if round(self.y0, round_decimals) < 0:
            exception_or_log_error(
                msg=f'{self} has negative y coordinate in {self.page}.',
                fail_loudly=str(validation) != str(BboxValidationTypes.DISABLED),
                exception_type=ValueError,
                handler=handler,
            )

        if round(self.x0, round_decimals) < 0:
            exception_or_log_error(
                msg=f'{self} has negative x coordinate in {self.page}.',
                fail_loudly=str(validation) != str(BboxValidationTypes.DISABLED),
                exception_type=ValueError,
                handler=handler,
            )

    def check_overlap(self, bbox: Union['Bbox', Dict]) -> bool:
        """Verify if there's overlap between two Bboxes."""
        if isinstance(bbox, dict) and (
            bbox['x0'] <= self.x1 and bbox['x1'] >= self.x0 and bbox['y0'] <= self.y1 and bbox['y1'] >= self.y0
        ):
            return True
        elif type(bbox) is type(self) and (
            bbox.x0 <= self.x1 and bbox.x1 >= self.x0 and bbox.y0 <= self.y1 and bbox.y1 >= self.y0
        ):
            return True
        else:
            return False

    @property
    def area(self):
        """Return area covered by the Bbox."""
        return round(abs(self.x0 - self.x1) * abs(self.y0 - self.y1), 3)

    @classmethod
    def from_image_size(cls, x0, x1, y0, y1, page: Page) -> 'Bbox':
        """
        Create a Bbox from image dimensions, based on the scaling of character Bboxes within the Document.

        This method computes the coordinates of the bottom-left and top-right corners in
        a coordinate system where the y-axis is oriented from bottom to top, the x-axis is from left to right,
        and the scale is based on the page.

        :param x0: The x-coordinate of the top-left corner in an image-scaled system.
        :param x1: The x-coordinate of the bottom-right corner in an image-scaled system.
        :param y0: The y-coordinate of the top-left corner in an image-scaled system.
        :param y1: The y-coordinate of the bottom-right corner in an image-scaled system.
        :param page: The Page object for reference in scaling.

        :return: A Bbox object with rescaled dimensions.
        """
        factor_y = page.height / page.image_height
        factor_x = page.width / page.image_width
        image_height = page.image_height

        temp_y0 = (image_height - y0) * factor_y
        temp_y1 = (image_height - y1) * factor_y
        y0 = temp_y1
        y1 = temp_y0
        x0 = x0 * factor_x
        x1 = x1 * factor_x

        return cls(x0=x0, x1=x1, y0=y0, y1=y1, page=page)

    @property
    def x0_image(self):
        """Get the x0 coordinate in the context of the Page image."""
        return self.x0 * (self.page.image_width / self.page.width)

    @property
    def x1_image(self):
        """Get the x1 coordinate in the context of the Page image."""
        return self.x1 * (self.page.image_width / self.page.width)

    @property
    def y0_image(self):
        """Get the y0 coordinate in the context of the Page image, in a top-down coordinate system."""
        return self.page.image_height - self.y1 * (self.page.image_height / self.page.height)

    @property
    def y1_image(self):
        """Get the y1 coordinate in the context of the Page image, in a top-down coordinate system."""
        return self.page.image_height - self.y0 * (self.page.image_height / self.page.height)

    @property
    def dict_format(self) -> Dict:
        """Obtain Bbox data as a dictionary."""
        return {
            'page_index': self.page.index,
            'top': self.page.height - self.y1,
            'bottom': self.page.height - self.y0,
            'x0': self.x0,
            'x1': self.x1,
            'y0': self.y0,
            'y1': self.y1,
        }


class AnnotationSet(Data):
    """
    An Annotation Set is a group of Annotations. The Labels of those Annotations refer to the same Label Set.

    For more details see https://dev.konfuzio.com/sdk/explanations.html#annotation-set-concept
    """

    def __init__(self, document, label_set: 'LabelSet', id_: Union[int, None] = None, **kwargs):
        """
        Create an Annotation Set.

        :param id: ID of the Annotation Set
        :param document: Document where the Annotation Set belongs
        :param label_set: Label set where the Annotation Set belongs to
        :param annotations: Annotations of the Annotation Set
        """
        self.id_local = next(Data.id_iter)
        self.id_ = id_
        self.label_set: LabelSet = label_set
        self.document: Document = document  # we don't add it to the Document as it's added via get_annotations
        self._force_offline = document._force_offline
        self._annotations = []
        document.add_annotation_set(self)

    def __repr__(self):
        """Return string representation of the Annotation Set."""
        return f'{self.__class__.__name__}({self.id_}) of {self.label_set} in {self.document}.'

    def __lt__(self, other: 'AnnotationSet'):
        """Sort AnnotationSets by their first Annotation."""
        self_annotations = self.annotations(use_correct=False, ignore_below_threshold=True)
        other_annotations = other.annotations(use_correct=False, ignore_below_threshold=True)
        return self_annotations[0] < other_annotations[0]

    def annotations(self, use_correct: bool = True, ignore_below_threshold: bool = False):
        """All Annotations currently in this Annotation Set."""
        if not self._annotations:
            for annotation in self.document.annotations(use_correct=False, ignore_below_threshold=False):
                if annotation.annotation_set == self:
                    self._annotations.append(annotation)

        annotations: List[Annotation] = []
        if use_correct:
            annotations = [ann for ann in self._annotations if ann.is_correct]
        elif ignore_below_threshold:
            annotations = [
                ann
                for ann in self._annotations
                if ann.is_correct or (ann.confidence and ann.confidence >= ann.label.threshold)
            ]
        else:
            annotations = self._annotations
        return annotations

    @property
    def is_default(self) -> bool:
        """Check if AnnotationSet is the default AnnotationSet of the Document."""
        return self.label_set.is_default

    @property
    def start_offset(self) -> Optional[int]:
        """Calculate the earliest start based on all Annotations above detection threshold in this AnnotationSet."""
        return min(
            (s.start_offset for a in self.annotations(use_correct=False, ignore_below_threshold=True) for s in a.spans),
            default=None,
        )

    @property
    def start_line_index(self) -> Optional[int]:
        """Calculate starting line of this Annotation Set."""
        if self.start_offset is None:
            return None
        return self.document.text[0 : self.start_offset].count('\n')

    @property
    def end_offset(self) -> Optional[int]:
        """Calculate the end based on all Annotations above detection threshold currently in this AnnotationSet."""
        return max(
            (a.end_offset for a in self.annotations(use_correct=False, ignore_below_threshold=True)), default=None
        )

    @property
    def end_line_index(self) -> Optional[int]:
        """Calculate ending line of this Annotation Set."""
        if self.end_offset is None:
            return None
        return self.document.text[0 : self.end_offset].count('\n')


class LabelSet(Data):
    """
    A Label Set is a group of Labels.

    For more details see https://dev.konfuzio.com/sdk/explanations.html#label-set-concept
    """

    def __init__(
        self,
        project,
        labels=None,
        id_: int = None,
        name: str = None,
        name_clean: str = None,
        is_default=False,
        categories=None,
        has_multiple_annotation_sets=False,
        **kwargs,
    ):
        """
        Create a named Label Set.

        :param project: Project where the Label Set belongs
        :param id_: ID of the Label Set
        :param name: Name of Label Set
        :param name_clean: Normalized name of the Label Set
        :param labels: Labels that belong to the Label Set (IDs)
        :param is_default: Bool for the Label Set to be the default one in the Project
        :param categories: Categories to which the Label Set belongs
        :param has_multiple_annotation_sets: Bool to allow the Label Set to have different Annotation Sets in a Document
        """
        if categories is None:
            categories = []
        if labels is None:
            labels = []
        self.id_local = next(Data.id_iter)
        self.id_ = id_
        self.name = name
        self.name_clean = name_clean
        self.is_default = is_default

        if not categories and 'default_label_sets' in kwargs:
            self._default_of_label_set_ids = kwargs['default_label_sets']
            self.categories = []
        elif not categories and 'default_section_labels' in kwargs:
            self._default_of_label_set_ids = kwargs['default_section_labels']
            self.categories = []
        elif isinstance(categories, list) and all(isinstance(category, dict) for category in categories):
            self._default_of_label_set_ids = [category['id'] for category in categories]
            self.categories = []
        else:
            self._default_of_label_set_ids = []
            self.categories = categories

        self.has_multiple_annotation_sets = has_multiple_annotation_sets

        if 'has_multiple_sections' in kwargs:
            self.has_multiple_annotation_sets = kwargs['has_multiple_sections']

        self.project: Project = project
        self._force_offline = project._force_offline
        self.labels: List[Label] = []

        # todo allow to create Labels either on Project or Label Set level, so they are (not) shared among Label Sets.
        for label in labels:
            if isinstance(label, int):
                label = self.project.get_label_by_id(id_=label)
            self.add_label(label)

        if self not in project._label_sets:
            project.add_label_set(self)
        for category in self.categories:
            if self not in category.label_sets:
                category.add_label_set(self)

    def __lt__(self, other: 'LabelSet'):
        """Sort Label Sets by name."""
        try:
            return self.name < other.name
        except TypeError:
            logger.error(f'Cannot sort {self} and {other}.')
            return False

    def __repr__(self):
        """Return string representation of the Label Set."""
        return f'LabelSet: {self.name} ({self.id_})'

    def add_category(self, category: 'Category'):
        """
        Add Category to the Label Set, if it does not exist.

        :param category: Category to add to the Label Set
        """
        if category not in self.categories:
            self.categories.append(category)
        else:
            raise ValueError(f'In {self} the {category} is a duplicate and will not be added.')

    def add_label(self, label):
        """
        Add Label to Label Set, if it does not exist.

        :param label: Label ID to be added
        """
        if label not in self.labels:
            self.labels.append(label)
            label.add_label_set(self)
        else:
            raise ValueError(f'In {self} the {label} is a duplicate and will not be added.')
        return self

    def get_target_names(self, use_separate_labels: bool):
        """Get target string name for Annotation Label classification."""
        targets = []
        for label in self.labels:
            if use_separate_labels:
                targets.append(self.name + '__' + label.name)
            else:
                targets.append(label.name)
        return targets


class Category(Data):
    """
    Group Documents in a Project.

    For more details see https://dev.konfuzio.com/sdk/explanations.html#category-concept
    """

    def __init__(self, project, id_: int = None, name: str = None, name_clean: str = None, *args, **kwargs):
        """Associate Label Sets to relate to Annotations."""
        self.id_local = next(Data.id_iter)
        self.id_ = id_
        self.name = name
        self.name_clean = name_clean
        self.project: Project = project
        self._force_offline = project._force_offline
        self.label_sets: List[LabelSet] = []
        self.project.add_category(category=self)
        self._exclusive_first_page_strings = None
        self._exclusive_span_tokenizer = None

    @property
    def labels(self):
        """Return the Labels that belong to the Category and its Label Sets."""
        labels = set()
        for label_set in self.label_sets:
            labels.update(label_set.labels)
        return list(labels)

    @property
    def fallback_name(self) -> str:
        """Turn the Category name to lowercase, remove parentheses along with their contents, and trim spaces."""
        parentheses_removed = re.sub(r'\([^)]*\)', '', self.name.lower()).strip()
        single_spaces = parentheses_removed.replace('  ', ' ')
        return single_spaces

    def documents(self):
        """Filter for Documents of this Category."""
        return [x for x in self.project.documents if x.category == self]

    def test_documents(self):
        """Filter for test Documents of this Category."""
        return [x for x in self.project.test_documents if x.category == self]

    def add_label_set(self, label_set):
        """Add Label Set to Category."""
        if label_set not in self.label_sets:
            self.label_sets.append(label_set)
        else:
            raise ValueError(f'In {self} the {label_set} is a duplicate and will not be added.')

    @property
    def default_label_set(self):
        """Get the default Label Set of the Category."""
        # Search for existing default LabelSet
        default_label_set = next((label_set for label_set in self.label_sets if label_set.is_default), None)
        # If not found, create a new default LabelSet
        if not default_label_set:
            default_label_set = LabelSet(
                id_=self.id_,
                project=self.project,
                name=self.name,
                name_clean=self.name_clean,
                is_default=True,
                categories=[self],
                has_multiple_annotation_sets=False,
            )
        return default_label_set

    def _collect_exclusive_first_page_strings(self, tokenizer):
        """
        Collect exclusive first-page string across the Documents within the Category.

        :param tokenizer: A tokenizer to re-tokenize Documents within the Category before gathering the strings.
        """
        cur_first_page_strings = []
        cur_non_first_page_strings = []
        for doc in self.documents():
            doc = deepcopy(doc)
            doc = tokenizer.tokenize(doc)
            for page in doc.pages():
                if page.number == 1:
                    cur_first_page_strings.append({span.offset_string for span in page.spans()})
                else:
                    cur_non_first_page_strings.append({span.offset_string for span in page.spans()})
            doc.delete()
        if cur_first_page_strings:
            true_first_page_strings = set.intersection(*cur_first_page_strings)
        else:
            true_first_page_strings = set()
        if cur_non_first_page_strings:
            true_not_first_page_strings = set.intersection(*cur_non_first_page_strings)
        else:
            true_not_first_page_strings = set()
        true_first_page_strings = true_first_page_strings - true_not_first_page_strings
        self._exclusive_first_page_strings = true_first_page_strings

    def exclusive_first_page_strings(self, tokenizer) -> set:
        """
        Return a set of strings exclusive for first Pages of Documents within the Category.

        :param tokenizer: A tokenizer to process Documents before gathering strings.
        """
        if self._exclusive_span_tokenizer is not None:
            if tokenizer != self._exclusive_span_tokenizer:
                logger.warning(
                    'Assigned tokenizer does not correspond to the one previously used within this instance.'
                    'All previously found exclusive first-page strings within each Category will be removed '
                    'and replaced with the newly generated ones.'
                )
                self._collect_exclusive_first_page_strings(tokenizer)
        if not self._exclusive_first_page_strings:
            self._collect_exclusive_first_page_strings(tokenizer)
        return self._exclusive_first_page_strings

    def __lt__(self, other: 'Category'):
        """Sort Categories by name."""
        try:
            return self.name < other.name
        except TypeError:
            logger.error(f'Cannot sort {self} and {other}.')
            return False

    def __repr__(self):
        """Return string representation of the Category."""
        return f'Category: {self.name} ({self.id_})'


class CategoryAnnotation(Data):
    """
    Annotate the Category of a Page.

    For more details see https://dev.konfuzio.com/sdk/explanations.html#category-annotation-concept
    """

    def __init__(
        self,
        category: Category,
        confidence: Optional[float] = None,
        page: Optional[Page] = None,
        document: Optional['Document'] = None,
        id_: Optional[int] = None,
    ):
        """
        Create a CategoryAnnotation and link it to a Document or to a specific Page in a Document.

        :param id_: ID of the CategoryAnnotation.
        :param category: The Category to annotate the Page with.
        :param page: The Page to be annotated. Only use when not providing a Document.
        :param document: The Document to be annotated. Only use when not providing a Page.
        :param confidence: Predicted confidence of the CategoryAnnotation.
        """
        self.id_local = next(Data.id_iter)
        self.id_ = id_
        self.category = category
        self.page = page
        self.document = document
        if page is not None:
            if (document is not None) and (page.document != document):
                raise ValueError(
                    f'The provided {page} comes from {page.document} but the provided {document} does not correspond. '
                    f'You can provide just the Document argument if this CategoryAnnotation is not linked to a Page, '
                    f'otherwise only provide the Page argument; the corresponding Document will be found automatically.'
                )
            self.document = self.page.document
        self._confidence = confidence
        # Call add_category_annotation to Page at the end, so all attributes for duplicate checking are available.
        if self.page is not None:
            self.page.add_category_annotation(self)

    def __repr__(self):
        """Return string representation."""
        return f'Category Annotation: ({self.category}, {self.confidence}) in {self.page or self.document}'

    def __eq__(self, other):
        """Define equality condition for CategoryAnnotations.

        A CategoryAnnotation is equal to another if both the linked Page and the predicted Category are the same.
        """
        return (other is not None) and (self.page == other.page) and (self.category == other.category)

    def set_revised(self) -> None:
        """Set this Category Annotation as revised by human, and thus the correct one for the linked Page."""
        if self.page is not None:
            self.page.set_category(self.category)
        elif self.document is not None:
            self.document.set_category(self.category)

    @property
    def confidence(self) -> float:
        """
        Get the confidence of this Category Annotation.

        If the confidence was not set, it means it was never predicted by an AI. Thus, the returned value will
        be 0, unless it was set by a human, in which case it defaults to 1.

        :return: Confidence between 0.0 and 1.0 included.
        """
        # if confidence is None it means it was never predicted by an AI
        if self._confidence is None:
            # if this CategoryAnnotation was added by a human then the confidence is 1
            if (self.page is not None) and (self.page._human_chosen_category_annotation == self):
                return 1.0
            elif (self.document is not None) and (self.document._category == self.category):
                return 1.0
            else:
                # otherwise there is no prediction and no human revision so the confidence is 0
                return 0.0
        return self._confidence


class Label(Data):
    """
    Group Annotations across Label Sets.

    For more details see https://dev.konfuzio.com/sdk/explanations.html#label-concept
    """

    def __init__(
        self,
        project: 'Project',
        id_: Union[int, None] = None,
        text: str = None,
        get_data_type_display: str = 'Text',
        text_clean: str = None,
        description: str = None,
        label_sets=None,
        has_multiple_top_candidates: bool = False,
        threshold: float = 0.1,
        *initial_data,
        **kwargs,
    ):
        """
        Create a named Label.

        :param project: Project where the Label belongs
        :param id_: ID of the Label
        :param text: Name of the Label
        :param get_data_type_display: Data type of the Label
        :param text_clean: Normalized name of the Label
        :param description: Description of the Label
        :param label_sets: Label Sets that use this Label
        """
        self.id_local = next(Data.id_iter)
        self.id_ = id_
        self.name = text
        self.name_clean = text_clean
        self.data_type = get_data_type_display
        self.description = description
        self.has_multiple_top_candidates = has_multiple_top_candidates
        self.threshold = threshold
        self.project: Project = project
        self._force_offline = project._force_offline
        project.add_label(self)

        self.label_sets = []
        for label_set in label_sets or []:
            label_set.add_label(self)

        # Regex features
        self._tokens = {}
        self.tokens_file_path = None
        self._regex = {}  # : List[str] = []
        # self._combined_tokens = None
        self.regex_file_path = os.path.join(self.project.regex_folder, f'{self.name_clean}.json5')
        self._evaluations = {}  # used to do the duplicate check on Annotation level

        self._has_multiline_annotations = None

    def __repr__(self):
        """Return string representation."""
        return f'Label: {self.name}'

    def __lt__(self, other: 'Label'):
        """Sort Spans by start offset."""
        try:
            return self.name < other.name
        except TypeError:
            logger.error(f'Cannot sort {self} and {other}.')
            return False

    def annotations(
        self, categories: List[Category], use_correct=True, ignore_below_threshold=False
    ) -> List['Annotation']:
        """Return related Annotations. Consider that one Label can be used across Label Sets in multiple Categories."""
        annotations = []
        for category in categories:
            for document in category.documents():
                for annotation in document.annotations(
                    label=self, use_correct=use_correct, ignore_below_threshold=ignore_below_threshold
                ):
                    annotations.append(annotation)

        if not annotations:
            logger.warning(f'{self} has no correct annotations.')

        return annotations

    def spans(self, categories: List[Category], use_correct=True, ignore_below_threshold=False) -> List['Span']:
        """Return all Spans belonging to an Annotation of this Label."""
        annotations = self.annotations(
            categories=categories, use_correct=use_correct, ignore_below_threshold=ignore_below_threshold
        )
        spans = [span for annotation in annotations for span in annotation.spans]
        return spans

    def has_multiline_annotations(self, categories: List[Category] = None) -> bool:
        """Return if any Label annotations are multi-line."""
        if categories is None and self._has_multiline_annotations is None:
            raise TypeError(
                "This value has never been computed. Please provide a value for keyword argument: 'categories'"
            )
        elif isinstance(categories, list):
            self._has_multiline_annotations = False
            for category in categories:
                for document in category.documents():
                    for annotation in document.annotations(label=self):
                        if len(annotation.spans) > 1:
                            self._has_multiline_annotations = True
                            return True

        return self._has_multiline_annotations

    def add_label_set(self, label_set: 'LabelSet'):
        """
        Add Label Set to label, if it does not exist.

        :param label_set: Label Set to add
        """
        if label_set not in self.label_sets:
            self.label_sets.append(label_set)
        else:
            raise ValueError(f'In {self} the {label_set} is a duplicate and will not be added.')

    def evaluate_regex(self, regex, category: Category, annotations: List['Annotation'] = None, regex_quality=0):
        """
        Evaluate a regex on Categories.

        Type of regex allows you to group regex by generality

        Example:
            Three Annotations about the birthdate in two Documents and one regex to be evaluated
            1.doc: "My was born on the 12th of December 1980, you could also say 12.12.1980." (2 Annotations)
            2.doc: "My was born on 12.06.1997." (1 Annotations)
            regex: dd.dd.dddd (without escaped characters for easier reading)
            stats:
            - total_correct_findings: 2
            - correct_label_annotations: 3
            - total_findings: 2 --> precision 100 %
            - num_docs_matched: 2
            - Project.documents: 2  --> Document recall 100%

        """
        evaluations = []
        documents = category.documents()

        for document in documents:
            # todo: potential time saver: make sure we did a duplicate check for the regex before we run the evaluation
            evaluation = document.evaluate_regex(regex=regex, label=self, annotations=annotations)
            evaluations.append(evaluation)

        total_findings = sum(evaluation['count_total_findings'] for evaluation in evaluations)
        num_docs_matched = sum(evaluation['doc_matched'] for evaluation in evaluations)
        correct_findings = [finding for evaluation in evaluations for finding in evaluation['correct_findings']]
        total_correct_findings = sum(evaluation['count_total_correct_findings'] for evaluation in evaluations)
        processing_times = [evaluation['runtime'] for evaluation in evaluations]

        try:
            annotation_precision = total_correct_findings / total_findings
        except ZeroDivisionError:
            annotation_precision = 0

        try:
            annotation_recall = total_correct_findings / len(self.spans(categories=[category]))
        except ZeroDivisionError:
            annotation_recall = 0

        try:
            document_recall = num_docs_matched / len(documents)
        except ZeroDivisionError:
            document_recall = 0

        try:
            f_score = 2 * (annotation_precision * annotation_recall) / (annotation_precision + annotation_recall)
        except ZeroDivisionError:
            f_score = 0

        assert 0 <= annotation_precision <= 1
        assert 0 <= annotation_recall <= 1
        assert 0 <= document_recall <= 1
        assert 0 <= f_score <= 1

        if documents:
            evaluation = {
                'regex': regex,
                'regex_len': len(regex),  # the longer the regex the more conservative it is to use
                'runtime': sum(processing_times) / len(processing_times),  # time to process the regex
                'annotation_recall': annotation_recall,
                'annotation_precision': annotation_precision,
                'f1_score': f_score,
                'document_recall': document_recall,
                'regex_quality': regex_quality,
                # other stats
                'correct_findings': correct_findings,
                'total_findings': total_findings,
                'total_annotations': len(self.annotations(categories=[category])),
                'num_docs_matched': num_docs_matched,
                'total_correct_findings': total_correct_findings,
            }
            correct_matches_per_document = {
                f'document_{evaluation["id"]}': evaluation['correct_findings'] for evaluation in evaluations
            }
            evaluation.update(correct_matches_per_document)  # add the matching info per document

            return evaluation
        else:
            return {}

    def base_regex(self, category: 'Category', annotations: List['Annotation'] = None) -> str:
        """Find the best combination of regex in the list of all regex proposed by Annotations."""
        if category.id_ in self._tokens:
            return self._tokens[category.id_]

        logger.info(f'Beginning base regex search for Label {self.name}.')

        if annotations is None:
            all_annotations = self.annotations(categories=[category])  # default is use_correct = True
        else:
            all_annotations = annotations

        evaluated_proposals = []
        for annotation in all_annotations:
            annotation_proposals = annotation.tokens()
            evaluated_proposals += annotation_proposals

        self._evaluations[category.id_] = evaluated_proposals

        try:
            best_proposals = get_best_regex(evaluated_proposals)
        except ValueError:
            logger.error(f'We cannot find regexes for {self} with a f_score > 0.')
            best_proposals = []

        label_regex_token = merge_regex(best_proposals)

        self._tokens[category.id_] = label_regex_token

        return label_regex_token

    def _find_regexes(
        self, annotations, label_regex_token, category: 'Category', max_findings_per_page=100
    ) -> List[str]:
        """Find regexes for the Label."""
        search = [1, 3, 5]
        regex_to_remove_groupnames = re.compile(r'<.*?>')
        regex_to_remove_groupnames_full = re.compile(r'\?P<.*?>')

        regex_made = []
        regex_found = set()

        for annotation in annotations:
            new_proposals = []
            annotation.document.spans(fill=True)
            for span in annotation.spans:
                before_reg_dict = {}
                after_reg_dict = {}
                for spacer in search:  # todo fix this search, so that we take regex token from other spans into account
                    before_regex = ''
                    bef_spacer = spacer * 3 if spacer > 1 else spacer
                    before_start_offset = span.start_offset - bef_spacer  # spacer**2
                    for before_span in annotation.document.spans(
                        fill=True, start_offset=before_start_offset, end_offset=span.start_offset
                    ):
                        if before_span.annotation.label is self.project.no_label:
                            to_rep_offset_string = before_span.annotation.document.text[
                                max(before_start_offset, before_span.start_offset) : before_span.end_offset
                            ]
                            before_regex += suggest_regex_for_string(to_rep_offset_string, replace_characters=True)
                        else:
                            base_before_regex = before_span.annotation.label.base_regex(category)
                            stripped_base_before_regex = re.sub(regex_to_remove_groupnames_full, '', base_before_regex)
                            before_regex += stripped_base_before_regex
                    before_reg_dict[spacer] = before_regex

                    after_regex = ''
                    after_end_offset = span.end_offset + spacer
                    for after_span in annotation.document.spans(
                        fill=True, start_offset=span.end_offset, end_offset=after_end_offset
                    ):
                        if after_span.annotation.label is self.project.no_label:
                            to_rep_offset_string = after_span.annotation.document.text[
                                after_span.start_offset : min(after_end_offset, after_span.end_offset)
                            ]
                            after_regex += suggest_regex_for_string(to_rep_offset_string, replace_characters=True)
                        else:
                            base_after_regex = after_span.annotation.label.base_regex(category)
                            stripped_base_after_regex = re.sub(regex_to_remove_groupnames_full, '', base_after_regex)
                            after_regex += stripped_base_after_regex

                    after_reg_dict[spacer] = after_regex

                    spacer_proposals = [
                        before_regex + label_regex_token + after_regex,
                        before_reg_dict[search[0]] + label_regex_token + after_regex,
                        before_regex + label_regex_token + after_reg_dict[search[0]],
                    ]

                    # check for duplicates
                    for proposal in spacer_proposals:
                        new_regex = re.sub(regex_to_remove_groupnames, '', proposal)
                        if new_regex not in regex_found:
                            if max_findings_per_page:
                                num_matches = len(regex_matches(regex=proposal, doctext=annotation.document.text))
                                if num_matches / (annotation.document.number_of_pages) < max_findings_per_page:
                                    new_proposals.append(proposal)
                                else:
                                    logger.info(
                                        f'Skip to evaluate regex {repr(proposal)} as it finds {num_matches} in\
                                                    {annotation.document}.'
                                    )
                            else:
                                new_proposals.append(proposal)
            for proposal in new_proposals:
                new_regex = re.sub(regex_to_remove_groupnames, '', proposal)
                if new_regex not in regex_found:
                    regex_made.append(proposal)
                    regex_found.add(new_regex)

        logger.info(
            f'For Label {self.name} we found {len(regex_made)} regex proposals for {len(annotations)} Annotations.'
        )
        return regex_made

    def find_regex(self, category: 'Category', max_findings_per_page=100) -> List[str]:
        """Find the best combination of regex for Label with before and after context."""
        all_annotations = self.annotations(categories=[category])  # default is use_correct = True
        if all_annotations == []:
            logger.error(f'We cannot find annotations for Label {self} and Category {category}.')
            return []
        label_regex_token = self.base_regex(category=category, annotations=all_annotations)
        found_regex = self._find_regexes(all_annotations, label_regex_token, category, max_findings_per_page)
        # todo replace by compare
        evaluations = [
            self.evaluate_regex(_regex_made, category=category, annotations=all_annotations)
            for _regex_made in found_regex
        ]

        logger.info(
            f'We compare {len(evaluations)} regex for {len(all_annotations)} correct Annotations for {category}.'
        )

        try:
            logger.info(f'Evaluate {self} for best regex.')
            best_regex = get_best_regex(evaluations)
        except ValueError:
            logger.exception(f'We cannot find regex for {self} with a f_score > 0.')
            best_regex = []

        if best_regex == []:
            best_regex = [label_regex_token]

        return best_regex

    def regex(self, categories: List[Category], update=False) -> Dict:
        """Calculate regex to be used in the Extraction AI."""
        # if not is_file(self.regex_file_path, raise_exception=False) or update:
        logger.info(f'Build regexes for Label {self.name}.')
        regex = {}
        for category in categories:
            if category.id_ not in self._regex or update:
                regex_category_file_path = os.path.join(
                    self.project.regex_folder, f'{category.name}_{self.name_clean}_tokens.json5'
                )
                if not is_file(regex_category_file_path, raise_exception=False) or update:
                    category_regex = self.find_regex(category=category)
                    if os.path.exists(self.project.regex_folder):
                        with open(regex_category_file_path, 'w') as f:
                            json.dump(category_regex, f, indent=2, sort_keys=True)
                else:
                    logger.info(f'Start loading existing regexes for Label {self.name}.')
                    with open(regex_category_file_path, 'r') as f:
                        category_regex = json.load(f)
                regex[category.id_] = category_regex
            else:
                regex[category.id_] = self._regex[category.id_]

        self._regex = regex

        logger.info(f'Regexes are ready for Label {self.name}.')

        categories_ids = [category.id_ for category in categories]
        return {k: v for k, v in self._regex.items() if k in categories_ids}

    def spans_not_found_by_tokenizer(self, tokenizer, categories: List[Category], use_correct=False) -> List['Span']:
        """Find Label Spans that are not found by a tokenizer."""
        spans_not_found = []
        label_annotations = self.annotations(categories=categories, use_correct=use_correct)
        for annotation in label_annotations:
            for span in annotation.spans:
                if not tokenizer.span_match(span):
                    spans_not_found.append(span)
        return spans_not_found

    def lose_weight(self):
        """Delete data of the instance."""
        super().lose_weight()
        self._evaluations = {}
        self._tokens = {}
        self._regex = {}

    def get_probable_outliers_by_regex(
        self, categories: List[Category], use_test_docs: bool = False, top_worst_percentage: float = 0.1
    ) -> List['Annotation']:
        """
        Get a list of Annotations that are identified by the least precise regular expressions.

        This method iterates over the list of Categories and Annotations within each Category, collecting all the
        regexes associated with them. It then evaluates these regexes and collects the top worst ones (i.e., those with
        the least True Positives). For each of these top worst regexes, it returns the Annotations found by them but
        not by the best regex for that label, potentially identifying them as outliers.

        To detect outlier Annotations with multi-Spans, the method iterates over all the multi-Span Annotations under
        the Label and checks each Span that was not detected by the aforementioned worst regexes. If it is not found by
        any other regex in the Project, the entire Annotation is considered a potential outlier.

        :param categories: A list of Category objects under which the search is conducted.
        :type categories: List[Category]
        :param use_test_docs: Indicates whether the evaluation of the regular expressions occurs on test Documents or
                            training Documents.
        :type use_test_docs: bool
        :param top_worst_percentage: A threshold for determining what percentage of the worst regexes' output to return.
        :type top_worst_percentage: float

        :return: A list of Annotation objects identified by the least precise regular expressions.
        :rtype: List[Annotation]
        """
        if use_test_docs:
            documents = self.project.test_documents
        else:
            documents = self.project.documents
        outliers = set()
        for category in categories:
            true_positives = {}
            all_annotations = [
                annotation for annotation in self.annotations(categories=[category]) if annotation.is_correct
            ]
            found_regex = self.regex(categories)
            for regex in found_regex[category.id_]:
                for document in documents:
                    if regex in true_positives.keys():
                        true_positives[regex] += document.evaluate_regex(regex, self)['count_correct_annotations']
                    else:
                        true_positives[regex] = document.evaluate_regex(regex, self)['count_correct_annotations']
            if not true_positives:
                logger.warning(f'No regex was found for {self} in {category}.')
            elif not sum(true_positives.values()):
                logger.warning(f'No resultative regexes found for {self} in {category}.')
            else:
                regexes_with_percentages = {k: v / sum(true_positives.values()) for k, v in true_positives.items()}
                sorted_regexes = dict(sorted(regexes_with_percentages.items(), key=lambda item: item[1]))
                regexes_with_cumulative = {}
                for cur_regex, pct in sorted_regexes.items():
                    if not regexes_with_cumulative:
                        regexes_with_cumulative[cur_regex] = pct
                        previous_key = cur_regex
                    else:
                        regexes_with_cumulative[cur_regex] = sorted_regexes[cur_regex] + sorted_regexes[previous_key]
                        previous_key = cur_regex
                filtered_regexes = {k: v for k, v in regexes_with_cumulative.items() if v <= top_worst_percentage}
                max_tps_regex = self.find_regex(category=category)[0]
                detected_by_worst_spans = set()
                detected_by_best_spans = set()
                for regex in filtered_regexes:
                    cur_annotations_worst = set()
                    cur_annotations_best = set()
                    for annotation in all_annotations:
                        text = annotation.document.text
                        matches = regex_matches(text, regex, keep_full_match=False)
                        for span in annotation.spans:
                            for span_match in matches:
                                span_match_offsets = (span_match['start_offset'], span_match['end_offset'])
                                if span_match_offsets == (span.start_offset, span.end_offset):
                                    cur_annotations_worst.add(annotation)
                                    detected_by_worst_spans.add(span)
                        matches = regex_matches(text, max_tps_regex, keep_full_match=False)
                        for span in annotation.spans:
                            for span_match in matches:
                                span_match_offsets = (span_match['start_offset'], span_match['end_offset'])
                                if span_match_offsets == (span.start_offset, span.end_offset):
                                    cur_annotations_best.add(annotation)
                                    detected_by_best_spans.add(span)
                    if len(cur_annotations_worst) not in range(
                        round(len(cur_annotations_best) * 0.5), round(len(cur_annotations_best) * 1.5)
                    ):
                        outliers.update(cur_annotations_worst - cur_annotations_best)
                    for annotation in cur_annotations_worst.union(cur_annotations_best):
                        if len(annotation.spans) > 1:
                            text = annotation.document.text
                            for span in annotation.spans:
                                if span not in detected_by_worst_spans.union(detected_by_best_spans):
                                    cur_regex = None
                                    if found_regex[category.id_]:
                                        for top_regex in found_regex[category.id_]:
                                            matches = regex_matches(
                                                text,
                                                top_regex,
                                                keep_full_match=False,
                                            )
                                            if matches:
                                                for match in matches:
                                                    span_match_offsets = (
                                                        match['start_offset'],
                                                        match['end_offset'],
                                                    )
                                                    if span_match_offsets == (span.start_offset, span.end_offset):
                                                        cur_regex = top_regex
                                                break
                                    if not cur_regex:
                                        outliers.add(annotation)
                                break
        outliers = list(outliers)
        return outliers

    def get_probable_outliers_by_confidence(
        self,
        evaluation_data,
        confidence: float = 0.5,
    ) -> List['Annotation']:
        """
        Get a list of Annotations with the lowest confidence.

        A method iterates over the list of Categories, returning the top N Annotations with the lowest confidence score.

        :param evaluation_data: An instance of the ExtractionEvaluation class that contains predicted confidence scores.
        :type evaluation_data: ExtractionEvaluation instance
        :param confidence: A level of confidence below which the Annotations are returned.
        :type confidence: float
        """
        outliers = []
        all_annotations = evaluation_data.data[
            (evaluation_data.data['label_id'] == self.id_)
            & (evaluation_data.data['confidence_predicted'] < confidence)
            & (evaluation_data.data['is_correct'])
        ]
        for idx, outlier in all_annotations.iterrows():
            if outlier['id_']:
                document_id = outlier['document_id']
                cur_annotation = self.project.get_document_by_id(document_id).get_annotation_by_id(outlier['id_'])
                outliers.append(cur_annotation)
        return outliers

    def get_probable_outliers_by_normalization(self, categories: List[Category]) -> List['Annotation']:
        """
        Get a list of Annotations that do not pass normalization by the data type.

        A method iterates over the list of Categories, returning the Annotations that do not fit into the data type of
        a Label (= have None returned in an attempt of the normalization by the Label's data type).

        :param categories: Categories under which the search is done.
        :type categories: List[Category]
        """
        outliers = set()
        for category in categories:
            for annotation in [
                annotation for annotation in self.annotations(categories=[category]) if annotation.is_correct
            ]:
                for span in annotation.spans:
                    normalized = normalize(span.offset_string, self.data_type)
                    if not normalized:
                        outliers.add(annotation)
        outliers = list(outliers)
        return outliers

    def get_probable_outliers(
        self,
        categories: List[Category],
        regex_search: bool = True,
        regex_worst_percentage: float = 0.1,
        confidence_search: bool = True,
        evaluation_data=None,
        normalization_search: bool = True,
    ) -> List['Annotation']:
        """
        Get a list of Annotations that are outliers.

        Outliers are determined by either of three logics or a combination of them applied: found by the worst regex,
        have the lowest confidence and/or are not normalizeable by the data type of a given Label.

        :param categories: Categories under which the search is done.
        :type categories: List[Category]
        :param regex_search: Enable search by top worst regexes.
        :type regex_search: bool
        :param regex_worst_percentage: A % of Annotations returned by the regexes.
        :type regex_worst_percentage: float
        :param confidence_search: Enable search by the lowest-confidence Annotations.
        :type confidence_search: bool
        :param normalization_search: Enable search by normalizing Annotations by the Label's data type.
        :type normalization_search: bool
        :raises ValueError: When all search options are disabled.
        """
        if not regex_search and not confidence_search and not normalization_search:
            raise ValueError('All search modes disabled, search is impossible. Enable at least one search mode.')
        results = []
        if regex_search:
            results.append(
                set(self.get_probable_outliers_by_regex(categories, top_worst_percentage=regex_worst_percentage))
            )
        if confidence_search:
            results.append(set(self.get_probable_outliers_by_confidence(evaluation_data)))
        if normalization_search:
            results.append(set(self.get_probable_outliers_by_normalization(categories)))
        intersection_results = list(set.intersection(*results))
        return intersection_results

    # def save(self) -> bool:
    #     """
    #     Save Label online.
    #
    #     If no Label Sets are specified, the Label is associated with the first default Label Set of the Project.
    #
    #     :return: True if the new Label was created.
    #     """
    #     if len(self.label_sets) == 0:
    #         prj_label_sets = self.project.label_sets
    #         label_set = [t for t in prj_label_sets if t.is_default][0]
    #         label_set.add_label(self)
    #
    #     response = create_label(
    #         project_id=self.project.id_,
    #         label_name=self.name,
    #         description=self.description,
    #         has_multiple_top_candidates=self.has_multiple_top_candidates,
    #         data_type=self.data_type,
    #         label_sets=self.label_sets,
    #     )
    #
    #     return True


class Span(Data):
    """
    A Span is a sequence of characters or whitespaces without line break.

    For more details see https://dev.konfuzio.com/sdk/explanations.html#span-concept
    """

    def __init__(
        self,
        start_offset: int,
        end_offset: int,
        annotation: 'Annotation' = None,
        document: 'Document' = None,
        strict_validation: bool = True,
    ):
        """
        Initialize the Span without bbox, to save storage.

        If Bbox should be calculated the bbox file of the Document will be automatically downloaded.

        :param start_offset: Start of the offset string (int)
        :param end_offset: Ending of the offset string (int)
        :param annotation: The Annotation the Span belongs to. If not set, the Span is considered "virtual"
        :param document: Document the Span belongs to. If not specified, the annotation document is used.
        :param strict_validation: Whether to apply strict validation rules.
        See https://dev.konfuzio.com/sdk/tutorials/data_validation/index.html
        """
        self.id_local = next(Data.id_iter)

        self.document: Document = document
        if annotation and document:
            assert annotation.document is document
        self.annotation: Annotation = annotation
        if annotation:
            self.document = annotation.document

        self.start_offset = start_offset
        self.end_offset = end_offset
        self.top = None
        self.bottom = None
        self._line_index = None
        self._page: Union[Page, None] = None
        self._bbox: Union[Bbox, None] = None
        self.regex_matching = []
        # check because we don't want to add a Span multiple times to the Annotation
        if annotation and self not in annotation.spans:  # only add if Span has access to an Annotation
            annotation.add_span(self)
        self._valid(strict_validation)

    def _valid(self, strict: bool = True, handler: str = 'sdk_validation'):
        """
        Validate containted data.

        :param strict: If False, it allows Spans to have zero length, or span more than one visual line. For more
        details see https://dev.konfuzio.com/sdk/tutorials/data_validation/index.html
        """
        if self.end_offset == self.start_offset == 0:
            logger.warning(f'{self} is intentionally left empty.')
        elif self.start_offset < 0 or self.end_offset < 0:
            exception_or_log_error(
                msg=f'{self} must span text.',
                fail_loudly=strict,
                exception_type=ValueError,
                handler=handler,
            )
        elif self.start_offset == self.end_offset:
            exception_or_log_error(
                msg=f'{self} must span text: Start {self.start_offset} equals end.',
                fail_loudly=strict,
                exception_type=ValueError,
                handler=handler,
            )
        elif self.end_offset < self.start_offset:
            exception_or_log_error(
                msg=f'{self} length must be positive.',
                fail_loudly=strict,
                exception_type=ValueError,
                handler=handler,
            )
        elif self.offset_string and ('\n' in self.offset_string or '\f' in self.offset_string):
            exception_or_log_error(
                msg=f'{self} must not span more than one visual line.',
                fail_loudly=strict,
                exception_type=ValueError,
                handler=handler,
            )
        return True

    @property
    def page(self) -> Page:
        """Return Page of Span."""
        if self.document is None:
            raise NotImplementedError
        elif self.document.text is None:
            logger.error(f'{self.document} does not provide text.')
            pass
        elif self._page is None and self.document.pages():
            text = self.document.text[: self.start_offset]
            page_index = len(text.split('\f')) - 1
            self._page = self.document.get_page_by_index(page_index=page_index)
        return self._page

    @property
    def line_index(self) -> int:
        """Return index of the line of the Span."""
        self._valid()
        if self.document.text and self._line_index is None:
            line_number = len(self.document.text[: self.start_offset].replace('\f', '\n').split('\n'))
            self._line_index = line_number - 1

        return self._line_index

    def __eq__(self, other) -> bool:
        """Twp Spans are equal if their start_offset and end_offset are both equal."""
        return (
            type(self) == type(other)
            and self.start_offset == other.start_offset
            and self.end_offset == other.end_offset
        )

    def __lt__(self, other: 'Span'):
        """We compare Spans by their start offset. If start offsets are equal, we use end offsets."""
        return (self.start_offset, self.end_offset) < (other.start_offset, other.end_offset)

    def __repr__(self):
        """Return string representation."""
        if self.offset_string and len(self.offset_string) < 16:
            offset_string_repr = self.offset_string
        elif self.offset_string:
            offset_string_repr = f'{self.offset_string[:14]}[...]'
        else:
            offset_string_repr = ''

        if not self.annotation:
            return f'Virtual {self.__class__.__name__} ({self.start_offset}, {self.end_offset}): "{offset_string_repr}"'
        else:
            return f'{self.__class__.__name__} ({self.start_offset}, {self.end_offset}): "{offset_string_repr}"'

    def __hash__(self):
        """Make any online or local concept hashable. See https://stackoverflow.com/a/7152650."""
        if not self.annotation:
            raise NotImplementedError('Span without Annotation is not hashable.')
        else:
            return hash((self.annotation, self.start_offset, self.end_offset))

    def regex(self):
        """Suggest a Regex for the offset string."""
        if self.annotation:
            # todo make the options to replace characters and string more granular
            full_replace = suggest_regex_for_string(self.offset_string, replace_characters=True, replace_numbers=True)
            return merge_regex([full_replace])
        else:
            raise NotImplementedError('A Span needs a Annotation and Document relation to suggest a Regex.')

    def bbox(self) -> Bbox:
        """Calculate the bounding box of a text sequence."""
        if not self.document:
            raise NotImplementedError
        if not self.page:
            logger.warning(f'{self} does not have a Page.')
            return None
        if not self.document.bboxes_available:
            logger.warning(f'{self.document} of {self} does not provide Bboxes.')
            return None
        _ = self.line_index  # quick validate if start and end is in the same line of text

        if self._bbox is None:
            character_range = range(self.start_offset, self.end_offset)
            document = self.document
            characters = {key: document.bboxes.get(key) for key in character_range if document.text[key] != ' '}
            if not all(characters.values()):
                logger.error(f"{self} in {self.document} contains Characters that don't provide a Bounding Box.")
            self._bbox = Bbox(
                x0=min([ch.x0 for c, ch in characters.items() if ch is not None]),
                x1=max([ch.x1 for c, ch in characters.items() if ch is not None]),
                y0=max(0, min([ch.y0 for c, ch in characters.items() if ch is not None])),
                y1=max([ch.y1 for c, ch in characters.items() if ch is not None]),
                page=self.page,
                validation=self.document._bbox_validation_type,
            )
        return self._bbox

    def bbox_dict(self) -> Dict:
        """Return Span Bbox info as a serializable Dict format for external integration with the Konfuzio Server."""
        span_dict = {
            'start_offset': self.start_offset,
            'end_offset': self.end_offset,
            'line_number': self.line_index + 1,
            'offset_string': self.offset_string,
            'offset_string_original': self.offset_string,
            'page_index': self.bbox().page.index,
            'top': self.bbox().page.height - self.bbox().y1,
            'bottom': self.bbox().page.height - self.bbox().y0,
            'x0': self.bbox().x0,
            'x1': self.bbox().x1,
            'y0': self.bbox().y0,
            'y1': self.bbox().y1,
        }
        return span_dict

    @property
    def normalized(self):
        """Normalize the offset string."""
        return normalize(self.offset_string, self.annotation.label.data_type)

    @property
    def offset_string(self) -> Union[str, None]:
        """Calculate the offset string of a Span."""
        if self.document and self.document.text:
            return self.document.text[self.start_offset : self.end_offset]
        else:
            return None

    def eval_dict(self):
        """Return any information needed to evaluate the Span."""
        if self.start_offset == self.end_offset == 0:
            span_dict = {
                'id_local': None,
                'id_': None,
                'confidence': None,
                'offset_string': None,
                'normalized': None,
                'start_offset': 0,  # to support compare function to evaluate True and False
                'end_offset': 0,  # to support compare function to evaluate True and False
                'is_correct': None,
                'created_by': None,
                'revised_by': None,
                'custom_offset_string': None,
                'revised': None,
                'label_threshold': None,
                'label_id': 0,
                'label_has_multiple_top_candidates': None,
                'label_set_id': 0,
                'annotation_id': None,
                'annotation_set_id': 0,  # to allow grouping to compare boolean
                'document_id': 0,
                'document_id_local': 0,
                'category_id': 0,
                'x0': 0,
                'x1': 0,
                'y0': 0,
                'y1': 0,
                'line_index': 0,
                'page_index': None,
                'page_width': 0,
                'page_height': 0,
                'x0_relative': None,
                'x1_relative': None,
                'y0_relative': None,
                'y1_relative': None,
                'page_index_relative': None,
                'area_quadrant_two': 0,
                'area': 0,
                'label_name': None,
                'label_set_name': None,
                'data_type': None,
            }
        else:
            span_dict = {
                'id_local': self.annotation.id_local,
                'id_': self.annotation.id_,
                'confidence': self.annotation.confidence,
                'offset_string': self.offset_string,
                'normalized': self.normalized,
                'start_offset': self.start_offset,  # to support multiline
                'end_offset': self.end_offset,  # to support multiline
                'is_correct': self.annotation.is_correct,
                'created_by': self.annotation.created_by,
                'revised_by': self.annotation.revised_by,
                'custom_offset_string': self.annotation.custom_offset_string,
                'revised': self.annotation.revised,
                'label_threshold': self.annotation.label.threshold,  # todo: allow to optimize threshold
                'label_id': self.annotation.label.id_,
                'label_has_multiple_top_candidates': self.annotation.label.has_multiple_top_candidates,
                'label_set_id': self.annotation.label_set.id_,
                'annotation_id': self.annotation.id_,
                'annotation_set_id': self.annotation.annotation_set.id_,
                'document_id': self.document.id_ if self.document.id_ else self.document.copy_of_id,
                'document_id_local': self.document.id_local,
                'category_id': self.document.category.id_,
                'line_index': self.line_index,
                'data_type': self.annotation.label.data_type,
            }

            if self.bbox():
                span_dict['x0'] = self.bbox().x0
                span_dict['x1'] = self.bbox().x1
                span_dict['y0'] = self.bbox().y0
                span_dict['y1'] = self.bbox().y1

                # https://www.cuemath.com/geometry/quadrant/
                span_dict['area_quadrant_two'] = self.bbox().x0 * self.bbox().y0
                span_dict['area'] = self.bbox().area

            if self.page:  # todo separate as eval_dict on Page level
                span_dict['page_index'] = self.page.index
                span_dict['page_width'] = self.page.width
                span_dict['page_height'] = self.page.height
                span_dict['x0_relative'] = self.bbox().x0 / self.page.width
                span_dict['x1_relative'] = self.bbox().x1 / self.page.width
                span_dict['y0_relative'] = self.bbox().y0 / self.page.height
                span_dict['y1_relative'] = self.bbox().y1 / self.page.height
                span_dict['page_index_relative'] = self.page.index / self.document.number_of_pages

            document_id = self.document.id_ if self.document.id_ is not None else self.document.copy_of_id
            span_dict['document_id'] = document_id
            span_dict['label_name'] = self.annotation.label.name if self.annotation.label else None
            span_dict['label_set_name'] = self.annotation.label_set.name if self.annotation.label_set else None

        return span_dict

    @staticmethod
    def get_sentence_from_spans(spans: Iterable['Span'], punctuation=None) -> List[List['Span']]:
        """Return a list of Spans corresponding to Sentences separated by Punctuation."""
        if punctuation is None:
            punctuation = {'.', '!', '?'}

        # get the sentence spans
        sentence_spans: List[List[Span]] = [[]]
        for span in spans:
            # get the text of the span
            span_text = span.offset_string

            # find the start and end offsets of each sentence in the span
            prev_sentence_start_offset = 0
            for index, char in enumerate(span_text):
                if char == ' ':
                    continue
                if char in punctuation:
                    sentence_start_offset = span.start_offset + prev_sentence_start_offset
                    sentence_end_offset = span.start_offset + index + 1
                    sentence_spans[-1].append(
                        Span(
                            start_offset=sentence_start_offset,
                            end_offset=sentence_end_offset,
                            document=span.page.document,
                        )
                    )
                    sentence_spans.append([])
                    prev_sentence_start_offset = index + 1

            if prev_sentence_start_offset < len(span_text):
                sentence_start_offset = span.start_offset + prev_sentence_start_offset
                sentence_end_offset = span.end_offset
                sentence_spans[-1].append(
                    Span(
                        start_offset=sentence_start_offset,
                        end_offset=sentence_end_offset,
                        document=span.page.document,
                    )
                )

        sentence_spans = [x for x in sentence_spans if len(x)]
        return sentence_spans


class AnnotationsContainer(dict):
    """
    Hold a collection of Annotations in an efficient way (dict of tuples).

    Provides methods to interact with the AnnotationsContainer as if it was a list.
    """

    def remove(self, obj):
        """Delete an Annotation from the container. Mimicks the similar method of the list."""
        # obj._spans might be a list as it might come from a previous SDK version
        if isinstance(obj._spans, list):
            key = (
                tuple(sorted((s.start_offset, s.end_offset) for s in obj._spans)),
                obj.label.name,
            )
        else:
            key = (tuple(sorted(obj._spans.keys())), obj.label.name)
        del self[key]

    def append(self, obj):
        """Add an Annotation to the container. Mimicks the similar method of the list."""
        # obj._spans might be a list as it might come from a previous SDK version
        if isinstance(obj._spans, list):
            key = (
                tuple(sorted((s.start_offset, s.end_offset) for s in obj._spans)),
                obj.label.name,
            )
        else:
            key = (tuple(sorted(obj._spans.keys())), obj.label.name)
        self[key] = obj


class Annotation(Data):
    """
    Hold information that a Label, Label Set and Annotation Set has been assigned to and combines Spans.

    For more details see https://dev.konfuzio.com/sdk/explanations.html#annotation-concept
    """

    def __init__(
        self,
        document: 'Document',
        annotation_set_id: Union[int, None] = None,  # support to init from API output
        annotation_set: Union[AnnotationSet, None] = None,  # support to init from API output
        label: Union[int, Label, None] = None,
        label_set_id: Union[None, int] = None,
        label_set: Union[None, LabelSet] = None,
        is_correct: bool = False,
        revised: bool = False,
        normalized=None,
        id_: int = None,
        spans=None,
        accuracy: float = None,
        confidence: float = None,
        created_by: int = None,
        revised_by: int = None,
        translated_string: str = None,
        custom_offset_string: bool = False,
        offset_string: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the Annotation.

        An Annotation can be created either using Spans or using Bboxes. Make sure that Bboxes supplied actually contain
        text inside; supplying empty Bboxes will cause an error.

        :param label: ID of the Annotation
        :param is_correct: If the Annotation is correct or not (bool)
        :param revised: If the Annotation is revised or not (bool)
        :param id_: ID of the Annotation (int)
        :param accuracy: Accuracy of the Annotation (float) which is the Confidence
        :param document: Document to annotate
        :param annotation: Annotation Set of the Document where the Label belongs
        :param label_set_text: Name of the Label Set where the Label belongs
        :param translated_string: Translated string
        :param custom_offset_string: String as edited by a user
        :param label_set_id: ID of the Label Set where the Label belongs
        """
        self.id_local = next(Data.id_iter)
        self.is_correct = is_correct
        self.revised = revised
        self.normalized = normalized
        self.translated_string = translated_string
        self.document = document
        self._force_offline = self.document._force_offline
        self.created_by = created_by
        self.revised_by = revised_by
        if custom_offset_string:
            self.custom_offset_string = offset_string
        else:
            self.custom_offset_string = None
        self.id_ = id_  # Annotations can have None id_, if they are not saved online and are only available locally
        self._spans: Dict = {}

        self._bbox = None

        if accuracy is not None:  # it's a confidence
            self.confidence = accuracy
        elif confidence is not None:
            self.confidence = confidence
        elif self.id_ is not None and accuracy is None:  # hotfix: it's an online Annotation crated by a human
            self.confidence = 1
        elif accuracy is None and confidence is None:
            self.confidence = None
        else:
            raise ValueError('Annotation has an id_ but does not provide a confidence.')

        if isinstance(label, int):
            self.label: Label = self.document.project.get_label_by_id(label)
        elif sdk_isinstance(label, Label):
            self.label: Label = label
        elif isinstance(label, str):
            self.label: Label = self.document.project.get_label_by_name(label)
        elif label is None:
            self.label = self.document.project.no_label
            logger.warning(f'{self} in {self.document} initialized without Label information. Using NO_LABEL.')
        else:
            raise ValueError(f'Cannot initialize with {label=}. Must be int, str or Label.')

        # make sure an Annotation Set is available
        if isinstance(annotation_set_id, int):
            self.annotation_set: AnnotationSet = self.document.get_annotation_set_by_id(annotation_set_id)
        elif sdk_isinstance(annotation_set, AnnotationSet):
            # it's a safe way to look up the Annotation Set first. Otherwise users can add Annotation Sets which
            # do not relate to the Document
            self.annotation_set: AnnotationSet = self.document.get_annotation_set_by_id(annotation_set.id_)
        else:
            # it will be set later, or an exception is raised at the end of the init
            self.annotation_set = None

        # if no label_set_id we check if is passed by section_label_id
        if label_set_id is None and kwargs.get('label_set_id') is not None:
            label_set_id = kwargs.get('label_set_id')

        # handles association to an Annotation Set if the Annotation belongs to a Category
        if isinstance(label_set_id, int):
            label_set: LabelSet = self.document.project.get_label_set_by_id(label_set_id)
        elif label_set is None and label_set_id is None and len(self.label.label_sets) == 1:
            label_set = self.label.label_sets[0]

        if sdk_isinstance(label_set, LabelSet):
            if self.annotation_set is not None:
                assert label_set == self.annotation_set.label_set, f'Conflicting Label Set information provided\
                    {label_set=} and {self.annotation_set.label_set=}'
            elif label_set.has_multiple_annotation_sets:
                raise ValueError(
                    f'Cannot assign {self} to AnnotationSet. {label_set} can have multiple Annotation Sets. '
                    f'Please provide the Annotation Set or AnnotationSet ID.'
                )
            else:
                matching_annotation_sets = [
                    ann_set for ann_set in self.document.annotation_sets() if ann_set.label_set == label_set
                ]
                if len(matching_annotation_sets) == 1:
                    annotation_set = matching_annotation_sets[0]
                elif len(matching_annotation_sets) == 0:
                    annotation_set = AnnotationSet(label_set=label_set, document=self.document)
                else:
                    raise ValueError(
                        f'Found multiple Annotation Sets for {label_set} in {self.document}. '
                        f'This should not happen because {label_set.has_multiple_annotation_sets=}.'
                    )
                self.annotation_set = annotation_set

        if self.annotation_set.label_set.id_:
            # for legacy reasons, we allow all Annotations to be in the NO_LABEL_SET Annotation Set
            assert (
                self.label in self.annotation_set.label_set.labels
            ), f'{self.label} is not in {self.annotation_set.label_set.labels} of {self.annotation_set}.'

        for span in spans or []:
            self.add_span(span)

        self.selection_bbox = kwargs.get('selection_bbox', None)
        if isinstance(self.selection_bbox, dict):
            page = self.document.get_page_by_index(self.selection_bbox['page_index'])
            x0, x1, y0, y1 = (
                self.selection_bbox['x0'],
                self.selection_bbox['x1'],
                self.selection_bbox['y0'],
                self.selection_bbox['y1'],
            )
            self.selection_bbox = Bbox(x0=x0, x1=x1, y0=y0, y1=y1, page=page)

        # a variable to access Document Bboxes in case an Annotation is Bbox-based
        self._document_bboxes = None

        # TODO START LEGACY to support multiline Annotations
        input_bbox_dicts = kwargs.get('bboxes', None)
        if input_bbox_dicts and len(input_bbox_dicts) > 0:
            self._add_annotation_from_bbox_dicts(input_bbox_dicts=input_bbox_dicts)
        elif (
            input_bbox_dicts is None
            and kwargs.get('start_offset', None) is not None
            and kwargs.get('end_offset', None) is not None
        ):
            # Legacy support for creating Annotations with a single offset
            bbox = kwargs.get('bbox', {})
            _ = Span(start_offset=kwargs.get('start_offset'), end_offset=kwargs.get('end_offset'), annotation=self)
            logger.warning(f'{self} is empty')

        self.top = None
        self.bottom = None
        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.y1 = None

        # todo: remove this Annotation single Bbox
        bbox = kwargs.get('bbox')
        if bbox:
            self.top = bbox.get('top')
            self.bottom = bbox.get('bottom')
            self.x0 = bbox.get('x0')
            self.x1 = bbox.get('x1')
            self.y0 = bbox.get('y0')
            self.y1 = bbox.get('y1')

        self.page_number = kwargs.get('page_number', None)
        # END LEGACY -

        # regex features
        self._tokens = []
        self._regex = None

        # Call add_annotation to document at the end, so all attributes for duplicate checking are available.
        self.document.add_annotation(self)

        if not self.document:
            raise NotImplementedError(f'{self} has no Document and cannot be created.')
        if not self.label_set:
            raise NotImplementedError(f'{self} has no Label Set and cannot be created.')
        if not self.annotation_set:
            raise NotImplementedError(f'{self} has no Annotation Set and cannot be created.')
        if not self.label:
            raise NotImplementedError(f'{self} has no Label and cannot be created.')
        if not self.spans:
            exception_or_log_error(
                msg=f'{self} has no Spans and cannot be created.',
                fail_loudly=self.document.project._strict_data_validation,
                exception_type=NotImplementedError,
            )

    def __repr__(self):
        """Return string representation."""
        if self.label and self.document:
            span_str = ', '.join(f'{x.start_offset, x.end_offset}' for x in self.spans)
            return f'Annotation ({self.get_link()}) {self.label.name} {span_str}'
        elif self.label:
            return f'Annotation ({self.get_link()}) {self.label.name} ({self.spans})'
        else:
            return f'Annotation ({self.get_link()}) without Label ({self.start_offset}, {self.end_offset})'

    def __eq__(self, other):
        """We compare an Annotation based on its Label, Label Sets if it's online otherwise on the id_local."""
        return (
            (self._spans.keys() == other._spans.keys())
            and self.label
            and other.label
            and (self.label == other.label)
            and self.document
            and other.document
            and (self.document == other.document)
        )

    def __lt__(self, other):
        """If we sort Annotations we do so by start offset."""
        return min(self._spans.keys()) < min(other._spans.keys())

    def __hash__(self):
        """Identity of Annotation that does not change over time."""
        return hash((self.id_local, self.document))

    @property
    def page(self) -> Page:
        """Return Page of Annotation."""
        return self.spans[0].page

    @property
    def is_multiline(self) -> int:
        """Calculate if Annotation spans multiple lines of text."""
        logger.error('We cannot calculate this. The indicator is unreliable.')
        return self.offset_string.count('\n')

    @property
    def normalize(self) -> str:
        """Provide one normalized offset string due to legacy."""
        logger.warning('You use normalize on Annotation Level which is legacy.')
        return normalize(self.offset_string, self.label.data_type)

    @property
    def start_offset(self) -> int:
        """Legacy: One Annotation can have multiple start offsets."""
        logger.warning('You use start_offset on Annotation Level which is legacy.')
        return min((sa[0] for sa in self._spans.keys()), default=None)

    @property
    def end_offset(self) -> int:
        """Legacy: One Annotation can have multiple end offsets."""
        logger.warning('You use end_offset on Annotation Level which is legacy.')
        return max((sa[1] for sa in self._spans.keys()), default=None)

    @property
    def offset_string(self) -> List[str]:
        """View the string representation of the Annotation."""
        if len(self.spans) > 1:
            logger.warning(f'You use offset string on {self} level which is legacy.')
        if not self.custom_offset_string and self.document.text:
            result = [span.offset_string for span in self.spans]
        elif self.custom_offset_string:
            result = self.custom_offset_string
        else:
            result = []
        return result

    @property
    def eval_dict(self) -> List[dict]:
        """Calculate the Span information to evaluate the Annotation."""
        return [span.eval_dict() for span in self.spans]

    @property
    def label_set(self) -> LabelSet:
        """Return Label Set of Annotation."""
        return self.annotation_set.label_set

    def _add_annotation_from_bbox_dicts(self, input_bbox_dicts: list) -> None:
        """
        Based on a list of dictionaries where each contains metadata for a bbox, this method adds Annotations to the Document.
        If the dictionary has the fields "start_offset" and "end_offset", the method creates a Span based on such fields.
        If not, it parses the bbox coordinates from the dictionary keys and based on them, looks for Spans
        that could be contained within such bbox. If no Span was found, a `KeyError` is raised.

        :param input_bbox_dicts: List of dictionaries, each containing metadata such as bbox coordinates
        or Span's start_offset & end_offset used to create an Annotation.
        Note: either (start_offset, end_offset) or (x0, x1, y0, y1, page_index) should be provided in each item of `input_bbox_dicts`.
        """
        for input_bbox_dict in input_bbox_dicts:
            if 'start_offset' in input_bbox_dict.keys() and 'end_offset' in input_bbox_dict.keys():
                Span(
                    start_offset=input_bbox_dict['start_offset'],
                    end_offset=input_bbox_dict['end_offset'],
                    annotation=self,
                )
            elif (
                'x0' in input_bbox_dict.keys()
                and 'x1' in input_bbox_dict.keys()
                and 'y0' in input_bbox_dict.keys()
                and 'y1' in input_bbox_dict.keys()
                and 'page_index' in input_bbox_dict.keys()
            ):
                page = self.document.get_page_by_index(input_bbox_dict['page_index'])
                if not self._document_bboxes:
                    self._document_bboxes = self.document.bbox_dict
                input_bbox_dict['top'] = page.height - input_bbox_dict['y1']
                input_bbox_dict['bottom'] = page.height - input_bbox_dict['y0']
                # checking that Bboxes provided as input contain text inside
                merged_bboxes = get_merged_bboxes(
                    doc_bbox=self._document_bboxes, bboxes=[input_bbox_dict], doc_text=self.document.text
                )
                for merged_bbox in merged_bboxes:
                    try:
                        span = Span(start_offset=merged_bbox['start_offset'], end_offset=merged_bbox['end_offset'])
                        self.add_span(span)
                        _ = Bbox(
                            page=self.document.get_page_by_index(merged_bbox['page_index']),
                            x0=merged_bbox['x0'],
                            x1=merged_bbox['x1'],
                            y0=merged_bbox['y0'],
                            y1=merged_bbox['y1'],
                        )
                    except KeyError:
                        raise KeyError(
                            f'Cannot add {self} because Bbox {input_bbox_dict} does not have any text inside and '
                            f'it is not possible create Spans from the selected area. Please provide Bboxes'
                            f'that have text inside them or Spans to create an Annotation.'
                        )
            else:
                raise ValueError(f'SDK cannot read Bbox of Annotation {self.id_} in {self.document}: {input_bbox_dict}')

    def add_span(self, span: Span):
        """Add a Span to an Annotation incl. a duplicate check per Annotation."""
        if (span.start_offset, span.end_offset) not in self._spans:
            # add the Span first to make sure to bea able to do a duplicate check
            self._spans[(span.start_offset, span.end_offset)] = span  # one Annotation can span multiple Spans
            if span.annotation is not None and self != span.annotation:
                raise ValueError(f'{span} should be added to {self} but relates to {span.annotation}.')
            else:
                span.annotation = self  # todo feature to link one Span to many Annotations
                span.document = self.document
        else:
            raise ValueError(f'In {self} the {span} is a duplicate and will not be added.')
        return self

    def get_link(self):
        """Get link to the Annotation in the SmartView."""
        if self.is_online:
            return get_annotation_view_url(self.id_)
        else:
            return None

    def save(self, label_set_id=None, annotation_set_id=None, document_annotations: list = None) -> bool:
        """
        Save Annotation online.

        If there is already an Annotation in the same place as the current one, we will not be able to save the current
        annotation.

        In that case, we get the id_ of the original one to be able to track it.
        The verification of the duplicates is done by checking if the offsets and Label match with any Annotations
        online.
        To be sure that we are comparing with the information online, we need to have the Document updated.
        The update can be done after the request (per annotation) or the updated Annotations can be passed as input
        of the function (advisable when dealing with big Documents or Documents with many Annotations).

        Specify label_set_id if you want to create an Annotation belonging to a new Annotation Set. Specify
        annotation_set_id if you want to add an Annotation to an existing Annotation Set. Do not specify both of them.

        :param document_annotations: Annotations in the Document (list)
        :return: True if new Annotation was created
        """
        if self.label == self.document.project.no_label:
            raise ValueError('You cannot save Annotations with Label NO_LABEL.')
        if self.document.category == self.document.project.no_category:
            raise ValueError(f'You cannot save Annotations of Documents with {self.document.category}.')
        new_annotation_added = False

        if self.is_online:
            raise ValueError(f'You cannot update Annotations once saved online: {self.get_link()}')
            # update_annotation(id_=self.id_, document_id=self.document.id_, project_id=self.project.id_)

        if not self.is_online:
            if self.spans:
                spans_bboxes = self.spans
            elif self.bboxes:
                spans_bboxes = self.bboxes
            else:
                raise ValueError('Cannot save an Annotation without Spans and Bboxes.')
            response = post_document_annotation(
                document_id=self.document.id_,
                label_id=self.label.id_,
                label_set_id=label_set_id,
                confidence=self.confidence,
                is_correct=self.is_correct,
                revised=self.revised,
                annotation_set_id=annotation_set_id,
                session=self.document.project.session,
                spans=spans_bboxes,
            )
            if response.status_code == 201:
                json_response = json.loads(response.text)
                self.id_ = json_response['id']
                new_annotation_added = True
            elif response.status_code == 403:
                logger.error(response.text)
                try:
                    if 'In one Project you cannot label the same text twice.' in response.text:
                        if document_annotations is None:
                            # get the Annotation
                            self.document.update()
                            document_annotations = self.document.annotations()
                        # get the id_ of the existing annotation
                        is_duplicated = False
                        for annotation in document_annotations:
                            if (
                                annotation.start_offset == self.start_offset
                                and annotation.end_offset == self.end_offset
                                and annotation.label == self.label
                            ):
                                logger.error(f'ID of annotation online: {annotation.id_}')
                                self.id_ = annotation.id_
                                is_duplicated = True
                                break

                        # if there isn't a perfect match, the current Annotation is considered incorrect
                        if not is_duplicated:
                            self.is_correct = False

                        new_annotation_added = False
                    else:
                        logger.exception(f'Unknown issue to create Annotation {self} in {self.document}')
                except KeyError:
                    logger.error(f'Not able to save Annotation online: {response}')
        return new_annotation_added

    def regex_annotation_generator(self, regex_list) -> List[Span]:
        """
        Build Spans without Labels by regexes.

        :return: Return sorted list of Spans by start_offset
        """
        spans: List[Span] = []
        for regex in regex_list:
            dict_spans = regex_matches(doctext=self.document.text, regex=regex)
            for offset in list({(x['start_offset'], x['end_offset']) for x in dict_spans}):
                try:
                    span = Span(start_offset=offset[0], end_offset=offset[1], annotation=self)
                    spans.append(span)
                except ValueError as e:
                    logger.error(str(e))
        spans.sort()
        return spans

    def token_append(self, new_regex, regex_quality: int):
        """Append token if it is not a duplicate."""
        category = self.document.category
        regex_to_remove_group_names = re.compile('<.*?>')
        previous_matchers = [re.sub(regex_to_remove_group_names, '', t['regex']) for t in self._tokens]
        found_for_label = [
            re.sub(regex_to_remove_group_names, '', t['regex']) for t in (self.label._evaluations.get(category.id_, []))
        ]
        new_matcher = re.sub(regex_to_remove_group_names, '', new_regex)
        if new_matcher not in previous_matchers + found_for_label:  # only run evaluation if the token is truly new
            evaluation = self.label.evaluate_regex(new_regex, regex_quality=regex_quality, category=category)
            self._tokens.append(evaluation)
            logger.debug(f'Added new regex Token {new_matcher}.')
        else:
            logger.debug(f'Annotation Token {repr(new_matcher)} or regex {repr(new_regex)} does exist.')

    def tokens(self) -> List[str]:
        """Create a list of potential tokens based on Spans of this Annotation."""
        if not self._tokens:
            for span in self.spans:
                # the original string, with harmonized whitespaces
                harmonized_whitespace = suggest_regex_for_string(span.offset_string, replace_numbers=False)
                regex_w = f'(?P<Label_{self.label.id_}_W_{self.id_}_{span.start_offset}>{harmonized_whitespace})'
                self.token_append(new_regex=regex_w, regex_quality=0)
                # the original string, with numbers replaced
                numbers_replaced = suggest_regex_for_string(span.offset_string)
                regex_n = f'(?P<Label_{self.label.id_}_N_{self.id_}_{span.start_offset}>{numbers_replaced})'
                self.token_append(new_regex=regex_n, regex_quality=1)
                # the original string, with characters and numbers replaced
                full_replacement = suggest_regex_for_string(span.offset_string, replace_characters=True)
                regex_f = f'(?P<Label_{self.label.id_}_F_{self.id_}_{span.start_offset}>{full_replacement})'
                self.token_append(new_regex=regex_f, regex_quality=2)
        return self._tokens

    # todo can we circumvent the combined tokens
    def regex(self):
        """Return regex of this Annotation."""
        return self.label.combined_tokens(categories=[self.document.category])

    def delete(self, delete_online: bool = True) -> None:
        """Delete Annotation.

        :param delete_online: Whether the Annotation is deleted online or only locally.
        """
        if self.document.is_online and delete_online:
            delete_document_annotation(
                annotation_id=self.id_, session=self.document.project.session, delete_from_database=True
            )
            self.document.update()
        else:
            try:
                del self.document._annotations[(tuple(sorted(self._spans.keys())), self.label.name)]
            except KeyError as e:
                logger.warning(
                    f'Could not delete annotation with key {(tuple(sorted(self._spans.keys())), self.label.name)} in {self.document}: {e}'
                )
                # See also: https://git.konfuzio.com/konfuzio/objectives/-/issues/12402

    def bbox(self) -> Bbox:
        """Get Bbox encompassing all Annotation Spans."""
        self._bbox = Bbox(
            x0=min([span.bbox().x0 for span in self.spans]),
            x1=max([span.bbox().x1 for span in self.spans]),
            y0=min([span.bbox().y0 for span in self.spans]),
            y1=max([span.bbox().y1 for span in self.spans]),
            page=self.page,
        )
        return self._bbox

    @property
    def spans(self) -> List[Span]:
        """Return default entry to get all Spans of the Annotation."""
        return [self._spans[k] for k in sorted(self._spans.keys())]

    @property
    def bboxes(self) -> List[Dict]:
        """Return the Bbox information for all Spans in serialized format.

        This is useful for external integration (e.g. Konfuzio Server)."
        """
        return [span.bbox_dict() for span in self.spans]

    def lose_weight(self):
        """Delete data of the instance."""
        super().lose_weight()
        self._tokens = []


class DocumentStatuses(Enum):
    """
    Define statuses of a Document's processing.

    For more details see https://dev.konfuzio.com/sdk/explanations.html#document-concept
    """

    QUEUING_FOR_OCR = 0
    OCR_IN_PROGRESS = 10
    QUEUING_FOR_EXTRACTION = 1
    EXTRACTION_IN_PROGRESS = 20
    QUEUING_FOR_CATEGORIZATION = 3
    CATEGORIZATION_IN_PROGRESS = 30
    QUEUING_FOR_SPLITTING = 4
    SPLITTING_IN_PROGRESS = 40
    WAITING_FOR_SPLITTING_CONFIRMATION = 41
    DONE = 2
    COULD_NOT_BE_PROCESSED = 111


class Document(Data):
    """
    Access the information about one Document, which is available online.

    For more details see https://dev.konfuzio.com/sdk/explanations.html#document-concept
    """

    def __init__(
        self,
        project: 'Project',
        id_: Union[int, None] = None,
        file_url: str = None,
        status: List[Union[int, str]] = None,
        data_file_name: str = None,
        is_dataset: bool = None,
        dataset_status: int = None,
        updated_at: str = None,
        assignee: int = None,
        category_template: int = None,  # fix for Konfuzio Server API, it's actually an ID of a Category
        category: Category = None,
        category_confidence: Optional[float] = None,
        category_is_revised: bool = False,
        text: str = None,
        bbox: dict = None,
        bbox_validation_type=None,
        pages: list = None,
        update: bool = False,
        copy_of_id: Union[int, None] = None,
        *args,
        **kwargs,
    ):
        """
        Create a Document and link it to its Project.

        :param id_: ID of the Document
        :param project: Project where the Document belongs to
        :param file_url: URL of the Document
        :param status: Status of the Document
        :param data_file_name: File name of the Document
        :param is_dataset: Is dataset or not. (bool)
        :param dataset_status: Dataset status of the Document (e.g. Training)
        :param updated_at: Updated information
        :param assignee: Assignee of the Document
        :param bbox: Bounding box information per character in the PDF (dict)
        :param bbox_validation_type: One of ALLOW_ZERO_SIZE (default), STRICT, or DISABLED. Also see the `Bbox` class.
        :param pages: List of page sizes.
        :param update: Annotations, Annotation Sets will not be loaded by default. True will load it from the API.
                        False from local files
        :param copy_of_id: ID of the Document that originated the current Document
        """
        self._no_label_annotation_set = None
        self.id_local = next(Data.id_iter)
        self.id_ = id_
        self.assignee = assignee
        self._annotations: {} = None
        self._annotation_sets: List[AnnotationSet] = None
        self.file_url = file_url
        self.is_dataset = is_dataset
        self.dataset_status = dataset_status
        self._update = update
        self.copy_of_id = copy_of_id

        # The following variables come from the Server API
        # self._category -> document level category from the "category" field
        # self._category_confidence -> document level category confidence from the "category_confidence" field
        # self.category_is_revised -> document level boolean flag from the "category_is_revised" field
        if project and category_template:
            self._category = project.get_category_by_id(category_template)
        elif category:
            self._category = category
        else:
            self._category = project.no_category
        self._category_confidence = category_confidence
        self.category_is_revised = category_is_revised

        if updated_at:
            self.updated_at = dateutil.parser.isoparse(updated_at)
        else:
            self.updated_at = None

        self.name = data_file_name
        self.status = status  # status of Document online
        self.project = project
        self._force_offline = project._force_offline
        project.add_document(self)  # check for duplicates by ID before adding the Document to the project

        # use hidden variables to store low volume information in instance
        self._text: str = text
        self._text_hash = None
        self._characters: Dict[int, Bbox] = None
        self._pages_char_bboxes = None
        self._bbox_hash = None
        self._bbox_json = bbox
        self.bboxes_available: bool = bool(self.is_online or self._bbox_json)
        self._bbox_validation_type = bbox_validation_type
        if bbox_validation_type is None:
            if self.project._strict_data_validation:
                self._bbox_validation_type = BboxValidationTypes.ALLOW_ZERO_SIZE
            else:
                self._bbox_validation_type = BboxValidationTypes.DISABLED
        self._pages: List[Page] = []
        self._n_pages = None
        self.text_encoded: List[int] = None

        # prepare local setup for Document
        if self.is_online:
            pathlib.Path(self.document_folder).mkdir(parents=True, exist_ok=True)
        self.annotation_file_path = os.path.join(self.document_folder, 'annotations.json5')
        self.annotation_set_file_path = os.path.join(self.document_folder, 'annotation_sets.json5')
        self.txt_file_path = os.path.join(self.document_folder, 'document.txt')
        self.pages_file_path = os.path.join(self.document_folder, 'pages.json5')
        self.bbox_file_path = os.path.join(self.document_folder, 'bbox.zip')
        self.bio_scheme_file_path = os.path.join(self.document_folder, 'bio_scheme.txt')

        bbox_file_exists = is_file(self.bbox_file_path, raise_exception=False)
        self.bboxes_available: bool = self.is_online or self._bbox_json or bbox_file_exists

        if pages:
            self.pages()  # create Page instances

    def __repr__(self):
        """Return the name of the Document incl. the ID."""
        if self.id_ is None:
            return f'Virtual Document {self.name} ({self.copy_of_id})'
        else:
            return f'Document {self.name} ({self.id_})'

    @property
    def ocr_ready(self):
        """Check if Document OCR is ready."""
        return self.text is not None

    def update_meta_data(
        self,
        assignee: int = None,
        category_template: int = None,
        category: Category = None,
        data_file_name: str = None,
        dataset_status: int = None,
        status: List[Union[int, str]] = None,
        **kwargs,
    ):
        """Update document metadata information."""
        self.assignee = assignee

        if self.project and category_template:
            self._category = self.project.get_category_by_id(category_template)
        elif category:
            self._category = category
        else:
            self._category = None

        self.name = data_file_name

        self.status = status

        self.dataset_status = dataset_status

    def save_meta_data(self):
        """Save local changes to Document metadata to server."""
        update_document_konfuzio_api(
            document_id=self.id_,
            file_name=self.name,
            dataset_status=self.dataset_status,
            assignee=self.assignee,
            session=self.project.session,
        )

    def save(self):
        """Save all local changes to Document to server."""
        self.save_meta_data()

        for annotation in self.annotations(use_correct=False):
            if not annotation.is_online:
                try:
                    annotation.save()
                except HTTPError as e:
                    logger.error(str(e))

    @classmethod
    def from_file(
        self,
        path: str,
        project: 'Project',
        dataset_status: int = 0,
        category_id: Optional[int] = None,
        callback_url: str = '',
        timeout: Optional[int] = None,
        sync: bool = True,
    ) -> 'Document':
        """
        Initialize Document from file with synchronous API call.

        This class method will wait for the document to be processed by the server
        and then return the new Document. This may take a bit of time. When uploading
        many Documents, it is advised to set the sync option to False method.

        :param path: Path to file to be uploaded
        :param project: If to filter by correct annotations
        :param dataset_status: Dataset status of the Document (None: 0 Preparation: 1 Training: 2 Test: 3 Excluded: 4)
        :param category_id: Category the Document belongs to (if unset, it will be assigned one by the server)
        :param callback_url: Callback URL receiving POST call once extraction is done
        :param timeout: Number of seconds to wait for response from the server
        :param sync: Whether to wait for the file to be processed by the server
        :return: New Document
        """
        get_file_type_and_extension(path)  # check if file is valid
        response = upload_file_konfuzio_api(
            path,
            project_id=project.id_,
            dataset_status=dataset_status,
            category_id=category_id,
            callback_url=callback_url,
            sync=sync,
            session=konfuzio_session(timeout=timeout),
        )
        response = response.json()
        new_document_id = response['id']

        if sync:
            status = [
                response['status_data'],
                [status.name for status in DocumentStatuses if status.value == response['status_data']][0],
            ]
            if status[0] == 2:
                logger.debug(f'Document status code {status[0]}: {status[1]}')
            else:
                logger.warning(f'Document status code {status[0]}: {status[1]}')
            assert project.id_ == response['project'], 'Project id_ of uploaded file does not match'
            document = Document(
                id_=new_document_id,
                project=project,
                update=True,
                category_template=category_id if category_id else response['category'],
                text=response['text'],
                status=status[0],  # we want a numeric representation of status, e.g. 2, not the textual, e.g. "Done"
                data_file_name=response['data_file_name'],
                file_url=response['file_url'],
                dataset_status=dataset_status,
            )
        else:
            document = Document(
                id_=new_document_id,
                project=project,
                update=True,
                category_template=category_id,
                status=[0],  # numeric representation of a status that means "queueing for ocr"
                data_file_name=response['data_file_name'],
                dataset_status=dataset_status,
            )

        return document

    @property
    def file_path(self):
        """Return path to file."""
        return os.path.join(self.document_folder, amend_file_name(self.name))

    @property
    def category_annotations(self) -> List[CategoryAnnotation]:
        """
        Collect Category Annotations and average confidence across all Pages.

        :return: List of Category Annotations, one for each Category.
        """
        category_annotations = []
        for category in self.project.categories:
            if category != self.project.no_category:
                confidence = 0
                for page in self.pages():
                    confidence += page.get_category_annotation(category).confidence
                confidence /= self.number_of_pages
                if (confidence == 0.0) and (category == self._category):
                    confidence = self._category_confidence
                category_annotation = CategoryAnnotation(category=category, document=self, confidence=confidence)
                category_annotations.append(category_annotation)
        return category_annotations

    @property
    def maximum_confidence_category_annotation(self) -> Optional[CategoryAnnotation]:
        """
        Get the human revised Category Annotation of this Document, or the highest confidence one if not revised.

        :return: The found Category Annotation, or None if not present.
        """
        if self.category != self.project.no_category:
            # there is a unique Category Annotation per Category associated to this Document
            # by construction in Document.category_annotations
            return [
                category_annotation
                for category_annotation in self.category_annotations
                if category_annotation.category == self._category
            ][0]
        category_annotation = sorted(self.category_annotations, key=lambda x: x.confidence)[-1]
        if category_annotation.confidence != 0.0:
            return category_annotation
        return None

    @property
    def maximum_confidence_category(self) -> Optional[Category]:
        """
        Get the human revised Category of this Document, or the highest confidence one if not revised.

        :return: The found Category, or None if not present.
        """
        if self.maximum_confidence_category_annotation is not None:
            return self.maximum_confidence_category_annotation.category
        return self.project.no_category

    @property
    def category(self) -> Category:
        """
        Return the Category of the Document.

        The Category of a Document is only defined as long as all Pages have the same Category. Otherwise, the Document
        should probably be split into multiple Documents with a consistent Category assignment within their Pages, or
        the Category for each Page should be manually revised.
        """
        if not self.pages():
            return self._category
        all_pages_have_same_category = len({page.category for page in self.pages()} - {self.project.no_category}) == 1
        if all_pages_have_same_category:
            self._category = self.pages()[0].category
        else:
            self._category = self.project.no_category
        return self._category

    def get_segmentation(self, timeout: Optional[int] = None, num_retries: Optional[int] = None) -> List:
        """
        Retrieve the segmentation results for the Document.

        :param timeout: Number of seconds to wait for response from the server.
        :param num_retries: Number of retries if the request fails.
        :return: A list of segmentation results for each Page in the Document.
        """
        document = self.project.get_document_by_id(self.copy_of_id) if self.copy_of_id else self
        if any(page._segmentation is None for page in document.pages()):
            document_id = document.id_
            detectron_document_results = get_results_from_segmentation(
                document_id, self.project.id_, session=konfuzio_session(timeout=timeout, num_retries=num_retries)
            )
            assert len(detectron_document_results) == self.number_of_pages
            for page_index, detectron_page_result in enumerate(detectron_document_results):
                document.get_page_by_index(page_index)._segmentation = detectron_page_result
        else:
            detectron_document_results = [page._segmentation for page in document.pages()]

        return detectron_document_results

    def set_category(self, category: Category) -> None:
        """Set the Category of the Document and the Category of all of its Pages as revised."""
        if not category:
            category = self.project.no_category
        logger.info(f'Setting Category of {self} to {category}.')
        if category not in [self._category, self.project.no_category] and (
            self._category and self._category.name != self.project.no_category.name
        ):
            raise ValueError(
                'We forbid changing Category when already existing, because this requires some validations that are '
                'currently implemented in the Konfuzio Server. We recommend changing the Category of a Document via '
                'the Konfuzio Server.'
            )
        for page in self.pages():
            page.set_category(category)
        self._category = category
        self.category_is_revised = True

    @property
    def ocr_file_path(self):
        """Return path to OCR PDF file."""
        return os.path.join(self.document_folder, amend_file_name(self.name, append_text='ocr', new_extension='.pdf'))

    @property
    def number_of_pages(self) -> int:
        """Calculate the number of Pages."""
        if self._n_pages is None:
            self._n_pages = len(self.text.split('\f'))
        return self._n_pages

    @property
    def number_of_lines(self) -> int:
        """Calculate the number of lines."""
        return len(self.text.replace('\f', '\n').split('\n'))

    @property
    def no_label_annotation_set(self) -> AnnotationSet:
        """
        Return the Annotation Set for project.no_label Annotations.

        We need to load the Annotation Sets from Server first (call self.annotation_sets()).
        If we create the no_label_annotation_set in the first place, the data from the Server is not be loaded
        anymore because _annotation_sets will no longer be None.
        """
        if self._no_label_annotation_set is None:
            self.annotation_sets()

            # Find no label annotation set
            self._no_label_annotation_set = next(
                (
                    annotation_set
                    for annotation_set in self._annotation_sets
                    if annotation_set.label_set == self.project.no_label_set
                ),
                None,
            )

            # Create no label annotation set if not found
            if self._no_label_annotation_set is None:
                self._no_label_annotation_set = AnnotationSet(document=self, label_set=self.project.no_label_set)

        return self._no_label_annotation_set

    def spans(
        self,
        label: Label = None,
        use_correct: bool = False,
        start_offset: int = 0,
        end_offset: int = None,
        fill: bool = False,
    ) -> List[Span]:
        """Return all Spans of the Document."""
        spans = {}

        annotations = self.annotations(
            label=label, use_correct=use_correct, start_offset=start_offset, end_offset=end_offset, fill=fill
        )

        for annotation in annotations:
            for span in annotation.spans:
                k = (span.start_offset, span.end_offset)
                if k not in spans:
                    spans[k] = span

        # if self.spans() == list(set(self.spans())):
        #     # todo deduplicate Spans. One text offset in a Document can ber referenced by many Spans of Annotations
        #     raise NotImplementedError

        return [spans[k] for k in sorted(spans.keys())]

    def eval_dict(self, use_view_annotations=False, use_correct=False, ignore_below_threshold=False) -> List[dict]:
        """Use this dict to evaluate Documents. The speciality: For every Span of an Annotation create one entry."""
        result = []
        if use_view_annotations:
            annotations = self.view_annotations()
        else:
            annotations = self.annotations(use_correct=use_correct, ignore_below_threshold=ignore_below_threshold)
        if not annotations:  # if there are no Annotations in this Documents
            result.append(Span(start_offset=0, end_offset=0).eval_dict())
        else:
            for annotation in annotations:
                result += annotation.eval_dict

        return result

    def check_bbox(self) -> None:
        """
        Run validation checks on the Document text and bboxes.

        This is run when the Document is initialized, and usually it's not needed to be run again because a Document's
        text and bboxes are not expected to change within the Konfuzio Server.

        You can run this manually instead if your pipeline allows changing the text or the bbox during the lifetime of
        a document. Will raise ValueError if the bboxes don't match with the text of the document, or if bboxes have
        invalid coordinates (outside page borders) or invalid size (negative width or height).

        This check is usually slow, and it can be made faster by calling Document.set_text_bbox_hashes() right after
        initializing the Document, which will enable running a hash comparison during this check.
        """
        warn('WIP: Modifications before the next stable release expected.', FutureWarning, stacklevel=2)
        if self._check_text_or_bbox_modified():
            self._characters = None
            _ = self.bboxes

    def __deepcopy__(self, memo) -> 'Document':
        """Create a new Document of the instance."""
        copy_id = self.id_ if self.id_ else self.copy_of_id
        document = Document(
            id_=None,
            project=self.project,
            category=self.category,
            text=self.text,
            copy_of_id=copy_id,
            bbox=self.get_bbox(),
        )
        for page in self.pages():
            copy_id = page.id_ if page.id_ else page.copy_of_id
            _ = Page(
                id_=None,
                document=document,
                start_offset=page.start_offset,
                end_offset=page.end_offset,
                copy_of_id=copy_id,
                number=page.number,
                original_size=(page.width, page.height),
                image_size=(page.image_width, page.image_height),
            )
            _.image_bytes = page.image_bytes
        return document

    def check_annotations(self, update_document: bool = False) -> bool:
        """Check if Annotations are valid - no duplicates and correct Category."""
        valid = True
        assignee = None

        try:
            self.get_annotations()

        except ValueError as error_message:
            valid = False

            if 'is a duplicate of' in str(error_message):
                logger.error(f'{self} has duplicated Annotations.')
                assignee = 1101  # duplicated-annotation@konfuzio.com

            elif 'related to' in str(error_message):
                logger.error(f'{self} has Annotations from an incorrect Category.')
                assignee = 1118  # category-issue@konfuzio.com

            else:
                raise ValueError('Error not expected.')

        if update_document and assignee is not None:
            # set the dataset status of the Document to Excluded
            update_document_konfuzio_api(
                document_id=self.id_,
                file_name=self.name,
                dataset_status=4,
                assignee=assignee,
                session=self.project.session,
            )

        return valid

    def annotation_sets(self, label_set: LabelSet = None) -> List[AnnotationSet]:
        """
        Return Annotation Sets of Documents.

        :param label_set: Label Set for which to filter the Annotation Sets.
        :return: Annotation Sets of Documents.
        """
        if self._annotation_sets is None:
            if self.is_online and not is_file(self.annotation_set_file_path, raise_exception=False):
                self.download_document_details()
            if is_file(self.annotation_set_file_path, raise_exception=False):
                with open(self.annotation_set_file_path, 'r') as f:
                    raw_annotation_sets = json.load(f)
                # first load all Annotation Sets before we create Annotations
                for raw_annotation_set in raw_annotation_sets:
                    # for backwards compatibility
                    if 'label_set' in raw_annotation_set.keys():
                        label_set_id = raw_annotation_set['label_set']['id']
                    elif 'section_label' in raw_annotation_set.keys():
                        label_set_id = raw_annotation_set['section_label']
                    _ = AnnotationSet(
                        id_=raw_annotation_set['id'],
                        document=self,
                        label_set=self.project.get_label_set_by_id(label_set_id),
                    )
            elif self._annotation_sets is None:
                self._annotation_sets = []  # Annotation sets cannot be loaded from Konfuzio Server
        annotation_sets = self._annotation_sets
        if label_set:
            # filter by Label Set given as argument
            annotation_sets = [
                annotation_set for annotation_set in annotation_sets if annotation_set.label_set == label_set
            ]
        return annotation_sets

    def annotations(
        self,
        label: Label = None,
        use_correct: bool = True,
        ignore_below_threshold: bool = False,
        start_offset: int = 0,
        end_offset: int = None,
        fill: bool = False,
    ) -> List[Annotation]:
        """
        Filter available annotations.

        :param label: Label for which to filter the Annotations.
        :param use_correct: If to filter by correct Annotations.
        :param ignore_below_threshold: To filter out Annotations with confidence below Label prediction threshold.
        :return: Annotations in the document.
        """
        self.get_annotations()
        start_offset = max(start_offset, 0)
        if end_offset:
            end_offset = min(end_offset, len(self.text))
        annotations: List[Annotation] = []
        add = False
        for annotation in self._annotations.values():
            # filter by correct information
            if not annotation.is_correct:
                if ignore_below_threshold and (
                    not annotation.confidence or annotation.confidence < annotation.label.threshold
                ):
                    continue
            for span in annotation.spans:
                if (use_correct and annotation.is_correct) or not use_correct:
                    # todo: add option to filter for overruled Annotations where mult.=F
                    # todo: add option to filter for overlapping Annotations, `add_annotation` just checks for identical
                    # filter by start and end offset, include Annotations that extend into the offset
                    if start_offset is not None and end_offset is not None:  # if the start and end offset are specified
                        latest_start = max(span.start_offset, start_offset)
                        earliest_end = min(span.end_offset, end_offset)
                        is_overlapping = latest_start - earliest_end < 0
                    else:
                        is_overlapping = True
                    if label is not None:  # filter by Label
                        if label == annotation.label and is_overlapping:
                            add = True
                    elif is_overlapping:
                        add = True

            # as multiline Annotations will be added twice
            if add:
                annotations.append(annotation)
                add = False

        if fill:
            # todo: we cannot assure that the Document has a Category, so Annotations must not require label_set
            spans = [range(span.start_offset, span.end_offset) for anno in annotations for span in anno.spans]
            if end_offset is None:
                end_offset = len(self.text)
            missings = get_missing_offsets(start_offset=start_offset, end_offset=end_offset, annotated_offsets=spans)

            for missing in missings:
                new_spans = []
                offset_text = self.text[missing.start : missing.stop]
                # we split Spans which span multiple lines, so that one Span comprises one line
                offset_of_offset = 0
                line_breaks = [
                    offset_line for offset_line in re.split(r'(\n|\f)', offset_text) if offset_line != ''
                ]  # , '\n', '\f'}]
                if not line_breaks:
                    continue
                for offset in line_breaks:
                    start = missing.start + offset_of_offset
                    offset_of_offset += len(offset)
                    end = missing.start + offset_of_offset
                    new_span = Span(start_offset=start, end_offset=end)
                    new_spans.append(new_span)

                new_annotation = Annotation(
                    document=self,
                    annotation_set=self.no_label_annotation_set,
                    label=self.project.no_label,
                    label_set=self.project.no_label_set,
                    spans=new_spans,
                )

                annotations.append(new_annotation)

        return sorted(annotations)

    def view_annotations(self, start_offset: int = 0, end_offset: int = None) -> List[Annotation]:
        """Get the best Annotations, where the Spans are not overlapping."""
        self.get_annotations()
        annotations: List[Annotation] = []

        filled = 0  # binary number keeping track of filled offsets
        priority_annotations = sorted(
            self.annotations(use_correct=False, start_offset=start_offset, end_offset=end_offset),
            key=lambda x: (
                not x.is_correct,  # x.is_correct == True first
                -x.confidence if x.confidence else 0,  # higher confidence first
                min([span.start_offset for span in x.spans]),
            ),
        )

        no_label_duplicates = set()  # for top Annotation filter
        for annotation in priority_annotations:
            if annotation.confidence is not None and annotation.label.threshold > annotation.confidence:
                continue
            if not annotation.is_correct and annotation.revised:  # if marked as incorrect by user
                continue
            if annotation.label is self.project.no_label:
                continue
            spans_num = 0
            for span in annotation.spans:
                for i in range(span.start_offset, span.end_offset):
                    spans_num |= 1 << i
            if spans_num & filled:
                # if there's overlap
                continue
            if (
                annotation.is_correct is False
                and annotation.label.has_multiple_top_candidates is False
                and (annotation.label.id_, annotation.annotation_set.id_) in no_label_duplicates
            ):
                continue
            annotations.append(annotation)
            filled |= spans_num
            if not annotation.label.has_multiple_top_candidates:
                no_label_duplicates.add((annotation.label.id_, annotation.annotation_set.id_))

        return sorted(annotations)

    def lose_weight(self):
        """Remove NO_LABEL, wrong and below threshold Annotations."""
        super().lose_weight()
        self._bbox_json = None
        self._characters = None
        for annotation in self.annotations(use_correct=False, ignore_below_threshold=False):
            if annotation.label is self.project.no_label:
                annotation.delete(delete_online=False)
            elif not annotation.is_correct and (
                not annotation.confidence or annotation.label.threshold > annotation.confidence or annotation.revised
            ):
                annotation.delete(delete_online=False)
            else:
                annotation.lose_weight()

    @property
    def document_folder(self):
        """Get the path to the folder where all the Document information is cached locally."""
        return os.path.join(self.project.documents_folder, str(self.id_))

    def get_file(self, ocr_version: bool = True, update: bool = False):
        """
        Get OCR version of the original file.

        :param ocr_version: Bool to get the ocr version of the original file
        :param update: Update the downloaded file even if it is already available
        :return: Path to the selected file.
        """
        if ocr_version:
            file_path = self.ocr_file_path
        else:
            file_path = self.file_path
        # backward compatibility check
        if isinstance(self.status, int) or not self.status:
            status = self.status
        else:
            status = self.status[0]
        if status == DocumentStatuses.DONE.value and (
            not file_path or not is_file(file_path, raise_exception=False) or update
        ):
            pdf_content = download_file_konfuzio_api(self.id_, ocr=ocr_version, session=self.project.session)
            with open(file_path, 'wb') as f:
                f.write(pdf_content)

        return file_path

    def get_images(self, update: bool = False):
        """
        Get Document Pages as PNG images.

        :param update: Update the downloaded images even they are already available
        :return: Path to PNG files.
        """
        return [page.get_image(update=update) for page in self.pages()]

    def download_document_details(self):
        """
        Retrieve data from a Document online in case Document has finished processing.

        Data includes Document's status, URL of its file, name of its file, date of
        las update, its text and pagination, Annotations and Annotation Sets; optionally,
        Category information.
        """
        if self.is_online:
            data = get_document_details(document_id=self.id_, session=self.project.session)
            self.status = data['status_data']
            self.file_url = data['file_url']
            self.name = data['data_file_name']
            self.updated_at = dateutil.parser.isoparse(data['updated_at'])
            if data['category']:
                self._category = self.project.get_category_by_id(data['category'])
            # write a file, even there are no annotations to support offline work
            annotations = get_document_annotations(document_id=self.id_, session=self.project.session)['results']
            with open(self.annotation_file_path, 'w') as f:
                json.dump(annotations, f, indent=2, sort_keys=True)

            with open(self.annotation_set_file_path, 'w') as f:
                json.dump(data['annotation_sets'], f, indent=2, sort_keys=True)

            with open(self.txt_file_path, 'w', encoding='utf-8') as f:
                if data['text']:
                    f.write(data['text'])

            with open(self.pages_file_path, 'w') as f:
                json.dump(data['pages'], f, indent=2, sort_keys=True)
        else:
            raise NotImplementedError

        return self

    def add_annotation(self, annotation: Annotation):
        """Add an Annotation to a Document.

        The Annotation is only added to the Document if the data validation tests are passing for this Annotation.
        See https://dev.konfuzio.com/sdk/tutorials/data_validation/index.html

        :param annotation: Annotation to add in the Document
        :return: Input Annotation.
        """
        if self._annotations is None:
            self.annotations()

        # Keep compatibility with older AI versions that use lists for spans
        if isinstance(annotation._spans, list):
            ann_key = (
                tuple(sorted((s.start_offset, s.end_offset) for s in annotation._spans)),
                annotation.label.name,
            )
        else:
            ann_key = (tuple(sorted(annotation._spans.keys())), annotation.label.name)

        duplicated = self._annotations.get(ann_key, None)

        if not duplicated:
            # Hotfix Text Annotation Server:
            #  Annotation belongs to a Label / Label Set that does not relate to the Category of the Document.
            # todo: add test that the Label and Label Set of an Annotation belong to the Category of the Document
            if self.category != self.project.no_category:
                if annotation.label_set is not None:
                    if annotation.label_set.categories:
                        if (self.category in annotation.label_set.categories) or (
                            annotation.label is self.project.no_label
                        ):
                            self._annotations[ann_key] = annotation
                        else:
                            exception_or_log_error(
                                msg=f'We cannot add {annotation} related to {annotation.label_set.categories} to {self}'
                                f' as the Document has {self.category}',
                                fail_loudly=self.project._strict_data_validation,
                                exception_type=ValueError,
                            )
                    else:
                        raise ValueError(f'{annotation} uses Label Set without Category, cannot be added to {self}.')
                else:
                    raise ValueError(f'{annotation} has no Label Set, which cannot be added to {self}.')
            else:
                if annotation.label is self.project.no_label and annotation.label_set is self.project.no_label_set:
                    self._annotations[ann_key] = annotation
                else:
                    raise ValueError(f'We cannot add {annotation} to {self} where the Category is {self.category}')
        else:
            exception_or_log_error(
                msg=f'In {self} the {annotation} is a duplicate of {duplicated} and will not be added.',
                fail_loudly=self.project._strict_data_validation,
                exception_type=ValueError,
            )

        return self

    def get_annotation_by_id(self, annotation_id: int) -> Annotation:
        """
        Return an Annotation by ID, searching within the Document.

        :param annotation_id: ID of the Annotation to get.
        """
        result = None
        if self._annotations is None:
            self.annotations()
        for annotation in self._annotations.values():
            if annotation.id_ == annotation_id:
                result = annotation
                break
        if result:
            return result
        else:
            raise IndexError(f'Annotation {annotation_id} is not part of {self}.')

    def add_annotation_set(self, annotation_set: AnnotationSet):
        """Add the Annotation Sets to the Document."""
        if annotation_set.document and annotation_set.document != self:
            raise ValueError('One Annotation Set must only belong to one Document.')
        if self._annotation_sets is None:
            self._annotation_sets = []
        if annotation_set not in self._annotation_sets:
            if annotation_set.label_set.has_multiple_annotation_sets is False:
                # if the Label Set does not allow multiple Annotation Sets, check if there is already an Annotation Set
                # with the same Label Set
                if annotation_set.label_set in [x.label_set for x in self._annotation_sets]:
                    logger.error(f'In {self} the {annotation_set.label_set} is already used by another Annotation Set.')
            self._annotation_sets.append(annotation_set)
        else:
            raise ValueError(f'In {self} the {annotation_set} is a duplicate and will not be added.')
        return self

    def get_annotation_set_by_id(self, id_: int) -> AnnotationSet:
        """
        Return an Annotation Set by ID.

        :param id_: ID of the Annotation Set to get.
        """
        result = None
        if self._annotation_sets is None:
            self.annotation_sets()
        for annotation_set in self._annotation_sets:
            if annotation_set.id_ == id_:
                result = annotation_set
        if result:
            return result
        else:
            raise IndexError(f'Annotation Set {id_} is not part of Document {self.id_}.')

    @property
    def default_annotation_set(self) -> AnnotationSet:
        """Return the default Annotation Set of the Document."""
        if self._annotation_sets is None:
            self.annotation_sets()
        for annotation_set in self._annotation_sets:
            if annotation_set.is_default:
                return annotation_set
        default_label_set = self.category.default_label_set
        default_annotation_set = AnnotationSet(
            id_=next(Data.id_iter),
            document=self,
            label_set=default_label_set,
        )
        return default_annotation_set

    def get_text_in_bio_scheme(self, update=False) -> List[Tuple[str, str]]:
        """
        Get the text of the Document in the BIO scheme.

        :param update: Update the bio annotations even they are already available
        :return: list of tuples with each word in the text and the respective label
        """
        converted_text = []
        if not is_file(self.bio_scheme_file_path, raise_exception=False) or update:
            converted_text = convert_to_bio_scheme(self)
            with open(self.bio_scheme_file_path, 'w', encoding='utf-8') as f:
                for word, tag in converted_text:
                    f.writelines(word + ' ' + tag + '\n')
                f.writelines('\n')
        else:
            with open(self.bio_scheme_file_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    if not line.strip():
                        continue
                    split_line = line.strip().split(' ')
                    word = split_line[0]
                    tag = ' '.join(split_line[1:])  # tag allowed to have multiple words
                    converted_text.append((word, tag))

        return converted_text

    def get_bbox(self) -> Dict:
        """
        Get bbox information per character of file. We don't store bbox as an attribute to save memory.

        :return: Bounding box information per character in the Document.
        """
        if self._bbox_json:
            bbox = self._bbox_json
        elif is_file(self.bbox_file_path, raise_exception=False):
            with zipfile.ZipFile(self.bbox_file_path, 'r') as archive:
                bbox = json.loads(archive.read('bbox.json5'))
        elif self.is_online and self.status and self.status == DocumentStatuses.DONE.value:
            # todo check for self.project.id_ and self.id_ and ?
            logger.info(f'Start downloading bbox files of {len(self.text)} characters for {self}.')
            bbox = get_document_bbox(document_id=self.id_, session=self.project.session)['bbox']
            # Use the `zipfile` module: `compresslevel` was added in Python 3.7
            with zipfile.ZipFile(
                self.bbox_file_path, mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as zip_file:
                # Dump JSON data
                dumped: str = json.dumps(bbox, indent=2, sort_keys=True)
                # Write the JSON data into `data.json` *inside* the ZIP file
                zip_file.writestr('bbox.json5', data=dumped)
                # Test integrity of compressed archive
                zip_file.testzip()
        else:
            self.bboxes_available = False
            bbox = {}

        return bbox

    def get_bbox_by_page(self, page_index: int) -> Dict[str, Dict]:
        """Return list of all bboxes in a Page."""
        if not self._pages_char_bboxes:
            self._pages_char_bboxes: List[Dict[str, Dict]] = [{} for _ in self.pages()]
            for char_index, bbox in self.get_bbox().items():
                bbox['char_index'] = int(char_index)
                # for backwards compatibility
                if 'page_index' in bbox.keys():
                    bbox_page_index = bbox['page_index']
                else:
                    bbox_page_index = bbox['page_number'] - 1
                self._pages_char_bboxes[bbox_page_index][char_index] = bbox
        return self._pages_char_bboxes[page_index]

    @property
    def _hashable_characters(self) -> Optional[frozenset]:
        """Convert bbox dict into a hashable type."""
        return frozenset(self._characters) if self._characters is not None else None

    def set_text_bbox_hashes(self) -> None:
        """Update hashes of Document text and bboxes. Can be used for checking later on if any changes happened."""
        self._text_hash = hash(self._text)
        self._bbox_hash = hash(self._hashable_characters)

    def _check_text_or_bbox_modified(self) -> bool:
        """Check if either the Document text or its bboxes have been modified in memory."""
        text_modified = self._text_hash != hash(self._text)
        bbox_modified = self._bbox_hash != hash(self._hashable_characters)
        return text_modified or bbox_modified

    @property
    def bboxes(self) -> Dict[int, Bbox]:
        """Use the cached bbox version."""
        if self.bboxes_available and self._characters is None:
            bbox = self.get_bbox()
            boxes = {}
            for character_index, box in bbox.items():
                x0 = box.get('x0')
                x1 = box.get('x1')
                y0 = box.get('y0')
                y1 = box.get('y1')
                if 'page_index' in box.keys():
                    page_index = box.get('page_index')
                else:
                    page_index = box.get('page_number') - 1
                page = self.get_page_by_index(page_index=page_index)
                box_character = box.get('text')
                document_character = self.text[int(character_index)]
                if box_character not in [' ', '\f', '\n'] and box_character != document_character:
                    exception_or_log_error(
                        msg=f'{self} Bbox provides Character "{box_character}" Document text refers to '
                        f'"{document_character}" with ID "{character_index}".',
                        fail_loudly=self.project._strict_data_validation,
                        exception_type=ValueError,
                    )
                boxes[int(character_index)] = Bbox(
                    x0=x0, x1=x1, y0=y0, y1=y1, page=page, validation=self._bbox_validation_type
                )
            self._characters = boxes
        return self._characters

    def set_bboxes(self, characters: Dict[int, Bbox]):
        """Set character Bbox dictionary."""
        characters = {int(key): bbox for key, bbox in characters.items()}

        for key, bbox in characters.items():
            bbox._valid(self._bbox_validation_type)

        self._characters = characters
        self.bboxes_available = True

    @property
    def text(self):
        """Get Document text. Once loaded stored in memory."""
        if self._text is not None:
            return self._text
        if self.is_online and not is_file(self.txt_file_path, raise_exception=False):
            self.download_document_details()
        if is_file(self.txt_file_path, raise_exception=False):
            with open(self.txt_file_path, 'r', encoding='utf-8') as f:
                self._text = f.read()
        return self._text

    def add_page(self, page: Page):
        """Add a Page to a Document."""
        if page not in self._pages:
            self._pages.append(page)
        else:
            raise ValueError(f'In {self} the {page} is a duplicate and will not be added.')

    def get_page_by_index(self, page_index: int):
        """Return the Page by index."""
        for page in self.pages():
            if page.index == page_index:
                return page
        raise IndexError(f'Page with Index {page_index} not available in {self}')

    def pages(self) -> List[Page]:
        """Get Pages of Document."""
        if self._pages:
            return self._pages
        if self.is_online and not is_file(self.pages_file_path, raise_exception=False):
            self.download_document_details()
            is_file(self.pages_file_path)
        if is_file(self.pages_file_path, raise_exception=False):
            with open(self.pages_file_path, 'r') as f:
                pages_data = json.loads(f.read())

            page_texts = self.text.split('\f')
            assert len(page_texts) == len(pages_data)
            start_offset = 0
            for page_index, page_data in enumerate(pages_data):
                page_text = page_texts[page_index]
                end_offset = start_offset + len(page_text)
                _ = Page(
                    id_=page_data['id'],
                    document=self,
                    number=page_data['number'],
                    original_size=page_data['original_size'],
                    image_size=page_data['size'],
                    start_offset=start_offset,
                    end_offset=end_offset,
                )
                start_offset = end_offset + 1

        return self._pages

    def update(self):
        """Update Document information."""
        self.delete_document_details()
        self.download_document_details()
        return self

    def delete_document_details(self):
        """Delete all local content information for the Document."""
        try:
            shutil.rmtree(self.document_folder)
        except FileNotFoundError:
            pass
        pathlib.Path(self.document_folder).mkdir(parents=True, exist_ok=True)
        self._annotations = None
        self._annotation_sets = None
        self._text = None
        self._pages = []

    def delete(self, delete_online: bool = False):
        """Delete Document."""
        self.project.del_document_by_id(self.id_, delete_online=delete_online)

    def evaluate_regex(self, regex, label: Label, annotations: List['Annotation'] = None):
        """Evaluate a regex based on the Document."""
        start_time = time.time()
        findings_in_document = regex_matches(
            doctext=self.text,
            regex=regex,
            keep_full_match=False,
            filtered_group=f'Label_{label.id_}',
            # filter by name of Label: one regex can match multiple Labels
        )
        processing_time = time.time() - start_time
        correct_findings = []

        label_annotations = self.annotations(label=label)

        label_spans_offsets = {
            (span.start_offset, span.end_offset): ann for ann in label_annotations for span in ann.spans
        }

        for finding in findings_in_document:
            key = (finding['start_offset'], finding['end_offset'])
            if key in label_spans_offsets:
                correct_findings.append(label_spans_offsets[key])

        try:
            annotation_precision = len(correct_findings) / len(findings_in_document)
        except ZeroDivisionError:
            annotation_precision = 0

        try:
            annotation_recall = len(correct_findings) / len(label_spans_offsets)
        except ZeroDivisionError:
            annotation_recall = 0

        try:
            f1_score = 2 * (annotation_precision * annotation_recall) / (annotation_precision + annotation_recall)
        except ZeroDivisionError:
            f1_score = 0

        assert 0 <= annotation_precision <= 1
        assert 0 <= annotation_recall <= 1
        assert 0 <= f1_score <= 1

        return {
            'id': self.id_local,
            'regex': regex,
            # processing_time time can vary slightly between runs, round to ms so that this variation does not affect
            # the choice of regexes when values are below ms and metrics are the same
            'runtime': round(processing_time, 3),
            'count_total_findings': len(findings_in_document),
            'count_total_correct_findings': len(correct_findings),
            'count_correct_annotations': len(label_annotations),
            'count_correct_annotations_not_found': len(correct_findings) - len(label_annotations),
            'doc_matched': len(correct_findings) > 0,
            'annotation_precision': annotation_precision,
            'document_recall': 0,  # keep this key to be able to use the function get_best_regex
            'annotation_recall': annotation_recall,
            'f1_score': f1_score,
            'correct_findings': correct_findings,
        }

    def get_annotations(self) -> List[Annotation]:
        """Get Annotations of the Document."""
        # if self.category is None:
        #    raise ValueError(f'Document {self} without Category must not have Annotations')

        annotation_file_exists = is_file(self.annotation_file_path, raise_exception=False)
        annotation_set_file_exists = is_file(self.annotation_set_file_path, raise_exception=False)

        if self._update or (self.is_online and (self._annotations is None or self._annotation_sets is None)):
            if self.is_online and (not annotation_file_exists or not annotation_set_file_exists or self._update):
                self.update()  # delete the meta of the Document details and download them again
                self._update = False  # Make sure we don't repeat to load once updated.

            self._annotation_sets = None  # clean Annotation Sets to not create duplicates
            self.annotation_sets()

            self._annotations = AnnotationsContainer()  # clean Annotations to not create duplicates
            # We read the annotation file that we just downloaded
            with open(self.annotation_file_path, 'r') as f:
                raw_annotations = json.load(f)

            if self.category == self.project.no_category:
                # for backwards compatibility
                # todo: not send no_label label to the server / see how it is handled now
                # they are ignored now
                no_label_raw_annotations = []
                for raw_annotation in raw_annotations:
                    # we want to ensure that both label files of old structure that have 'label_text' field and label
                    # files of a new structure that have 'label' field that is a dict with nested elements are properly
                    # parsed
                    if 'label_text' in raw_annotation.keys():
                        if raw_annotation['label_text'] == 'NO_LABEL':
                            no_label_raw_annotations.append(raw_annotation)
                    elif isinstance(raw_annotation['label'], dict):
                        if raw_annotation['label']['name'] == 'NO_LABEL':
                            no_label_raw_annotations.append(raw_annotation)

                raw_annotations = no_label_raw_annotations

            if raw_annotations:
                for raw_annotation in raw_annotations:
                    # remove the 'annotation_set' because we specify its id explicitly when creating the Annotation
                    # conditions for backward compatibility
                    if 'annotation_set' in raw_annotation:
                        raw_annotation['annotation_set_id'] = raw_annotation.pop('annotation_set')
                    else:
                        raw_annotation['annotation_set_id'] = raw_annotation.pop('section')
                    # same with the 'label_set_id'
                    if 'label_set' in raw_annotation:
                        raw_annotation['label_set_id'] = raw_annotation.pop('label_set')['id']
                    else:
                        raw_annotation['label_set_id'] = raw_annotation.pop('section_label_id')
                    # same with 'document'
                    raw_annotation.pop('document', None)
                    if isinstance(raw_annotation['label'], int):
                        label = self.project.get_label_by_id(id_=raw_annotation['label'])
                    else:
                        label = self.project.get_label_by_id(id_=raw_annotation['label']['id'])
                    # same with 'label'
                    raw_annotation.pop('label', None)
                    # same with 'span'/'bboxes'
                    if 'span' in raw_annotation:
                        raw_spans = raw_annotation['span']
                        raw_annotation.pop('span', None)
                    else:
                        raw_spans = raw_annotation['bboxes']
                    spans = [
                        Span(start_offset=span['start_offset'], end_offset=span['end_offset']) for span in raw_spans
                    ]
                    # same with 'annotation_set'
                    if (
                        raw_annotation['annotation_set_id']
                        and not self.project.get_label_set_by_id(
                            raw_annotation['label_set_id']
                        ).has_multiple_annotation_sets
                    ):
                        raw_annotation.pop('annotation_set_id', None)
                    _ = Annotation(document=self, id_=raw_annotation['id'], label=label, spans=spans, **raw_annotation)
                self._update = False  # Make sure we don't repeat to load once loaded.

        if self._annotations is None:
            self.annotation_sets()
            self._annotations = AnnotationsContainer()
            # We load the annotation file if it exists
            if annotation_file_exists:
                with open(self.annotation_file_path, 'r') as f:
                    raw_annotations = json.load(f)

                if self.category == self.project.no_category:
                    # conditions for backward compatibility
                    raw_annotations = [
                        annotation
                        for annotation in raw_annotations
                        if (
                            ('label_text' in annotation.keys() and annotation['label_text'] == 'NO_LABEL')
                            or (
                                'label' in annotation.keys()
                                and isinstance(annotation['label'], dict)
                                and annotation['label']['name'] == 'NO_LABEL'
                            )
                        )
                    ]

                if raw_annotations:
                    for raw_annotation in raw_annotations:
                        # for backward compatibility - to enable loading projects with old and new structure of folders
                        # and files / json5 metadata
                        if 'annotation_set' in raw_annotation:
                            raw_annotation['annotation_set_id'] = raw_annotation.pop('annotation_set')
                        else:
                            raw_annotation['annotation_set_id'] = raw_annotation.pop('section')
                        if 'label_set' in raw_annotation:
                            raw_annotation['label_set_id'] = raw_annotation.pop('label_set')['id']
                        else:
                            raw_annotation['label_set_id'] = raw_annotation.pop('section_label_id')
                        raw_annotation.pop('document', None)
                        if isinstance(raw_annotation['label'], int):
                            label = self.project.get_label_by_id(id_=raw_annotation['label'])
                        else:
                            label = self.project.get_label_by_id(id_=raw_annotation['label']['id'])
                        # same with 'label'
                        raw_annotation.pop('label', None)
                        # same with 'span'/'bboxes'
                        if 'span' in raw_annotation:
                            raw_spans = raw_annotation['span']
                            raw_annotation.pop('span', None)
                        else:
                            raw_spans = raw_annotation['bboxes']
                        spans = [
                            Span(start_offset=span['start_offset'], end_offset=span['end_offset']) for span in raw_spans
                        ]
                        # same with 'annotation_set'
                        if (
                            raw_annotation['annotation_set_id']
                            and not self.project.get_label_set_by_id(
                                raw_annotation['label_set_id']
                            ).has_multiple_annotation_sets
                        ):
                            raw_annotation.pop('annotation_set_id', None)
                        _ = Annotation(
                            document=self, id_=raw_annotation['id'], label=label, spans=spans, **raw_annotation
                        )

        return self._annotations.values()

    def propose_splitting(self, splitting_ai) -> List:
        """Propose splitting for a multi-file Document.

        :param splitting_ai: An initialized SplittingAI class
        """
        proposed = splitting_ai.propose_split_documents(self)
        return proposed

    def create_subdocument_from_page_range(self, start_page: Page, end_page: Page, include=False):
        """
        Create a shorter Document from a Page range of an initial Document.

        :param start_page: A Page that the new sub-Document starts with.
        :type start_page: Page
        :param end_page: A Page that the new sub-Document ends with, if include is True.
        :type end_page: Page
        :param include: Whether end_page is included into the new sub-Document.
        :type include: bool
        :returns: A new sub-Document.
        """
        if include:
            pages_text = self.text[start_page.start_offset : end_page.end_offset]
        else:
            pages_text = self.text[start_page.start_offset : end_page.start_offset]
        new_doc = Document(project=self.project, id_=None, text=pages_text, category=self.category)
        i = 1
        start_offset = 0
        for page in self.pages():
            end_offset = start_offset + len(page.text)
            page_id = page.id_ if page.id_ else page.copy_of_id
            if (include and page.number in range(start_page.number, end_page.number + 1)) or (
                not include and page.number in range(start_page.number, end_page.number)
            ):
                new_page = Page(
                    id_=None,
                    original_size=(page.height, page.width),
                    document=new_doc,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    copy_of_id=page_id,
                    number=i,
                )
                for category_annotation in page.category_annotations:
                    CategoryAnnotation(
                        category=category_annotation.category,
                        confidence=category_annotation.confidence,
                        page=new_page,
                    )
                i += 1
                start_offset = end_offset + 1
        return new_doc

    def get_page_by_id(self, page_id: int, original: bool = False) -> Page:
        """
        Get a Page by its ID.

        :param page_id: An ID of the Page to fetch.
        :type page_id: int
        """
        for page in self.pages():
            if page.id_ == page_id:
                return page
            if original:
                raise IndexError(f'Page id {page_id} was not found in {self}.')
            else:
                if not page.id_ and page.copy_of_id == page_id:
                    logger.warning(f'Page id {page_id} was not found in {self}, only a Page copy.')
                    return page

    @property
    def bbox_dict(self) -> Dict:
        """Get a dictionary of Document's character-level Bboxes."""
        return {
            key: {
                'x0': value['x0'],
                'x1': value['x1'],
                'y0': value['y0'],
                'y1': value['y1'],
                'page_index': value['page_index'],
                'page_number': value['page_index'] + 1,
                'line_index': value['line_index'],
                'text': value['text'],
                'top': self.get_page_by_index(value['page_index']).height - value['y1'],
                'bottom': self.get_page_by_index(value['page_index']).height - value['y0'],
            }
            for key, value in self.get_bbox().items()
        }


class Project(Data):
    """
    Access the information of a Project.

    For more details see https://dev.konfuzio.com/sdk/explanations.html#project-concept
    """

    def __init__(
        self,
        id_: Union[int, None],
        project_folder=None,
        update=False,
        max_ram=None,
        strict_data_validation: bool = True,
        credentials: dict = {},
        **kwargs,
    ):
        """
        Set up the Data using the Konfuzio Host.

        :param id_: ID of the Project
        :param project_folder: Set a Project root folder, if empty "data_<id_>" will be used.
        :param update: Whether to sync local files with the Project online.
        :param max_ram: Maximum RAM used by AI models trained on this Project.
        :param strict_data_validation: Whether to apply strict data validation rules.
        :param credentials: A dict of key/values that are available in the Project.
        See https://dev.konfuzio.com/sdk/tutorials/data_validation/index.html
        """
        self.id_local = next(Data.id_iter)
        self.id_ = id_  # A Project with None ID is not retrieved from the HOST
        if self.id_ is None:
            self.set_offline()
        self._project_folder = project_folder
        self.categories: List[Category] = []
        self._label_sets: List[LabelSet] = []
        self._labels: List[Label] = []
        self._documents: List[Document] = []
        self._meta_data = []
        self._max_ram = max_ram
        self._strict_data_validation = strict_data_validation
        self.credentials = credentials

        for key, value in kwargs.items():
            setattr(self, key, value)

        # paths
        self.meta_file_path = os.path.join(self.project_folder, 'documents_meta.json5')
        self.categories_and_label_data_file_path = os.path.join(self.project_folder, 'categories_and_label_data.json5')
        # labels and label sets left for backwards compatibility
        self.labels_file_path = os.path.join(self.project_folder, 'labels.json5')
        self.label_sets_file_path = os.path.join(self.project_folder, 'label_sets.json5')

        if self.id_ or self._project_folder:
            self.get(update=update)
        else:
            self.no_category = Category(project=self, id_=0, name_clean='NO_CATEGORY', name='NO_CATEGORY')
        # todo: list of Categories related to NO LABEL SET can be outdated, i.e. if the number of Categories changes
        self.no_label_set = LabelSet(
            project=self, id_=0, categories=self.categories, has_multiple_annotation_sets=False
        )
        self.no_label_set.name_clean = 'NO_LABEL_SET'
        self.no_label_set.name = 'NO_LABEL_SET'
        self.no_label = Label(project=self, id_=0, text='NO_LABEL', label_sets=[self.no_label_set])
        if not self.no_category.label_sets:
            self.no_category.add_label_set(self.no_label_set)
        if not self.no_label_set.categories:
            self.no_label_set.add_category(self.no_category)
        self.no_label.name_clean = 'NO_LABEL'
        self._regexes = None

    def __repr__(self):
        """Return string representation."""
        return f'Project {self.id_}'

    @property
    def ai_models(self):
        """Return all AIs."""
        return get_all_project_ais(project_id=self.id_, session=self.session)

    @property
    def documents(self):
        """Return Documents with status training."""
        return [doc for doc in self._documents if doc.dataset_status == 2]

    @property
    def online_documents_dict(self) -> Dict:
        """Return a dictionary of online documents using their id as key."""
        return {doc.id_: doc for doc in self._documents if isinstance(doc.id_, int)}

    @property
    def virtual_documents(self):
        """Return Documents created virtually."""
        return [doc for doc in self._documents if doc.dataset_status is None or doc.id_ is None]

    @property
    def test_documents(self):
        """Return Documents with status test."""
        return [doc for doc in self._documents if doc.dataset_status == 3]

    @property
    def excluded_documents(self):
        """Return Documents which have been excluded."""
        return [doc for doc in self._documents if doc.dataset_status == 4]

    @property
    def preparation_documents(self):
        """Return Documents with status test."""
        return [doc for doc in self._documents if doc.dataset_status == 1]

    @property
    def no_status_documents(self):
        """Return Documents with no status."""
        return [doc for doc in self._documents if doc.dataset_status == 0]

    @property
    def project_folder(self) -> str:
        """Calculate the data document_folder of the Project."""
        if self._project_folder is not None:
            return self._project_folder
        else:
            return f'data_{self.id_}'

    @property
    def regex_folder(self) -> str:
        """Calculate the regex folder of the Project."""
        return os.path.join(self.project_folder, 'regex')

    @property
    def documents_folder(self) -> str:
        """Calculate the regex folder of the Project."""
        return os.path.join(self.project_folder, 'documents')

    @property
    def model_folder(self) -> str:
        """Calculate the model folder of the Project."""
        return os.path.join(self.project_folder, 'models')

    @property
    def max_ram(self):
        """Return maximum memory used by AI models."""
        return self._max_ram

    def write_project_files(self):
        """Overwrite files with Project, Label, Label Set information."""
        categories_labels_label_sets = get_project_categories(project_id=self.id_, session=self.session)
        with open(self.categories_and_label_data_file_path, 'w') as f:
            json.dump(categories_labels_label_sets, f, indent=2, sort_keys=True)

        self.write_meta_of_files()

        return self

    def write_meta_of_files(self):
        """Overwrite meta-data of Documents in Project."""
        meta_data = get_meta_of_files(project_id=self.id_, session=self.session)
        with open(self.meta_file_path, 'w') as f:
            json.dump(meta_data, f, indent=2, sort_keys=True)

    def get(self, update=False):
        """
        Access meta information of the Project.

        :param update: Update the downloaded information even it is already available
        """
        pathlib.Path(self.project_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.documents_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.regex_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.model_folder).mkdir(parents=True, exist_ok=True)

        # reload means re-add to the Project what is currently available in the Project's folder
        # update means download information from server to keep up with the updates made online
        if self.id_ and (not is_file(self.meta_file_path, raise_exception=False) or update):
            self.write_project_files()
        # order of adding data into the project has changed after migration from v2 to v3 API - the endpoint used for
        # fetching categories, labels and label sets returns them in a nested structure:
        # Categories -> Label Sets -> Labels.
        # order of methods is pertained for backward compatibility in case there is project information that is stored
        # in an "old" way, with labels.json5 and label_sets.json5
        self.get_labels(reload=True)
        self.get_label_sets(reload=True)
        self.get_categories(reload=True)
        self.get_meta(reload=True)
        self.init_or_update_document(from_online=False)
        return self

    def add_label_set(self, label_set: LabelSet):
        """
        Add Label Set to Project, if it does not exist.

        :param label_set: Label Set to add in the Project
        """
        if label_set not in self._label_sets:
            self._label_sets.append(label_set)
        else:
            raise ValueError(f'In {self} the {label_set} is a duplicate and will not be added.')

    def add_category(self, category: Category):
        """
        Add Category to Project, if it does not exist.

        :param category: Category to add in the Project
        """
        if category.name != 'NO_CATEGORY':
            if category not in self.categories:
                self.categories.append(category)
            else:
                raise ValueError(f'In {self} the {category} is a duplicate and will not be added.')

    def add_label(self, label: Label):
        """
        Add Label to Project, if it does not exist.

        :param label: Label to add in the Project
        """
        if label not in self._labels:
            self._labels.append(label)
        else:
            raise ValueError(f'In {self} the {label} is a duplicate and will not be added.')

    def add_document(self, document: Document):
        """Add Document to Project, if it does not exist."""
        if document not in self._documents:
            self._documents.append(document)
        else:
            raise ValueError(f'In {self} the {document} is a duplicate and will not be added.')

    def get_meta(self, reload=False):
        """
        Get the list of all Documents in the Project and their information.

        :return: Information of the Documents in the Project.
        """
        if not self._meta_data or reload:
            if self._meta_data:
                self.old_meta_data = self._meta_data
            with open(self.meta_file_path, 'r') as f:
                self._meta_data = json.load(f)

        return self._meta_data

    @property
    def meta_data(self):
        """Return Project meta data."""
        if not self._meta_data:
            self.get_meta()
        return self._meta_data

    def get_categories(self, reload: bool = True):
        """Load Categories for all Label Sets in the Project."""

        # backward compatibility loading
        if is_file(self.label_sets_file_path, raise_exception=False):
            with open(self.label_sets_file_path, 'r') as f:
                label_sets_data = json.load(f)

            self.categories = []  # clean up Labels to not create duplicates

            # adding a NO_CATEGORY at this step because we need to preserve it after Project is updated
            if 'NO_CATEGORY' not in [category.name for category in self.categories]:
                self.no_category = Category(project=self, id_=0, name_clean='NO_CATEGORY', name='NO_CATEGORY')
            for label_set_data in label_sets_data:
                label_set = self.get_label_set_by_id(label_set_data['id'])
                if label_set.is_default:
                    category = Category(project=self, id_=label_set_data['id'], **label_set_data)
                    category.label_sets.append(label_set)
                    label_set.categories.append(category)  # Konfuzio Server mixes the concepts, we use two instances
            for label_set in self.label_sets:
                if label_set.is_default:
                    # the _default_of_label_set_ids are the Label Sets used by the Category
                    pass
                else:
                    # the _default_of_label_set_ids are the Categories the Label Set is used in
                    for label_set_id in label_set._default_of_label_set_ids:
                        category = self.get_category_by_id(label_set_id)
                        if category not in label_set.categories:
                            label_set.add_category(category)  # The Label Set is linked to a Category it created
                        if label_set.name not in [cur_label_set.name for cur_label_set in category.label_sets]:
                            category.add_label_set(label_set)

        if reload and (
            not is_file(self.labels_file_path, raise_exception=False)
            and not is_file(self.label_sets_file_path, raise_exception=False)
        ):
            self.categories = []

            if 'NO_CATEGORY' not in [category.name for category in self.categories]:
                self.no_category = Category(project=self, id_=0, name_clean='NO_CATEGORY', name='NO_CATEGORY')
            with open(self.categories_and_label_data_file_path, 'r') as f:
                categories_and_label_data = json.load(f)
            for category in categories_and_label_data:
                cur_category = Category(project=self, id_=category['id'], name=category['name'])
                for label_data in category['schema']:
                    cur_labels = label_data['labels']
                    label_data.pop('labels', None)
                    label_set = LabelSet(project=self, id_=label_data['id'], **label_data)
                    for cur_label in cur_labels:
                        if cur_label['name'] not in [label.name for label in self.labels]:
                            _ = Label(project=self, id_=cur_label['id'], text=cur_label['name'], **cur_label)
                        else:
                            _ = self.get_label_by_id(cur_label['id'])
                        label_set.labels.append(_)
                        _.label_sets.append(label_set)
                    if label_set.id_ == cur_category.id_:
                        label_set.is_default = True
                    cur_category.label_sets.append(label_set)
                    label_set.categories.append(cur_category)

    def get_label_sets(self, reload=False):
        """Get LabelSets in the Project."""
        if not self._label_sets or reload:
            self._label_sets = []  # clean up Label Sets to not create duplicates
            if is_file(self.label_sets_file_path, raise_exception=False):
                with open(self.label_sets_file_path, 'r') as f:
                    label_sets_data = json.load(f)
                for label_set_data in label_sets_data:
                    _ = LabelSet(project=self, id_=label_set_data['id'], **label_set_data)
                    if _.name not in [label_set.name for label_set in self._label_sets]:
                        self._label_sets.append(_)
            else:
                for cur_label_set in [label_set for category in self.categories for label_set in category.label_sets]:
                    self._label_sets.append(cur_label_set)
        return self._label_sets

    @property
    def label_sets(self):
        """Return Project LabelSets."""
        if not self._label_sets:
            self.get_label_sets()
        return self._label_sets

    def get_labels(self, reload=False) -> List[Label]:
        """Get ID and name of any Label in the Project."""
        if not self._labels or reload:
            self._labels = []  # clean up Labels to not create duplicates
            if is_file(self.labels_file_path, raise_exception=False):
                with open(self.labels_file_path, 'r') as f:
                    labels_data = json.load(f)
                for label_data in labels_data:
                    # Remove the Project from label_data
                    label_data.pop('project', None)
                    Label(project=self, id_=label_data['id'], **label_data)
            else:
                for cur_label in [
                    label
                    for category in self.categories
                    for label_set in category.label_sets
                    for label in label_set.labels
                ]:
                    if cur_label not in self._labels:
                        self._labels.append(cur_label)
        return self._labels

    @property
    def labels(self):
        """Return Project Labels."""
        if not self._labels:
            self.get_labels()
        return self._labels

    def init_or_update_document(self, from_online=False):
        """
        Initialize or update Documents from local files to then decide about full, incremental or no update.

        :param from_online: If True, all Document metadata info is first reloaded with latest changes in the server
        """
        logger.info(f'Running init_or_update_document({from_online=}) on {self}')
        local_docs_dict = self.online_documents_dict
        if from_online:
            self.write_meta_of_files()
            self.get_meta(reload=True)
        updated_docs_ids_set = set()  # delete all docs not in the list at the end
        n_updated_documents = 0
        n_new_documents = 0
        n_unchanged_documents = 0
        for document_data in self.meta_data:
            updated_docs_ids_set.add(document_data['id'])
            # if document_data['status'][0] == 2:  # - hotfix for Text Annotation Server # todo add test

            new_date = document_data['updated_at']
            updated = False
            new = document_data['id'] not in local_docs_dict
            if not new:
                last_date = local_docs_dict[document_data['id']].updated_at
                updated = dateutil.parser.isoparse(new_date) > last_date if last_date is not None else True
            category_id = None
            # backward compatibility
            if 'category' in document_data.keys():
                category_id = document_data['category']
            elif 'category_template' in document_data.keys():
                category_id = document_data['category_template']
            doc_category = self.get_category_by_id(category_id) if category_id else self.no_category
            document_data.pop('category', None)
            # ensuring we store document status in a variable and don't pass it multiple times by not removing it from
            # document_data
            status = document_data['status_data']
            document_data.pop('status_data', None)
            if updated:
                doc = local_docs_dict[document_data['id']]
                doc.update_meta_data(category=doc_category, status=status, **document_data)
                doc.update()
                logger.debug(f'{doc} was updated, we will download it again as soon you use it.')
                n_updated_documents += 1
            elif new:
                document_data.pop('project', None)
                # ensuring we store dataset status in a variable and don't pass it multiple times by not removing it
                # from document_data
                if 'status' in document_data:
                    status = document_data['status']
                document_data.pop('status', None)
                doc = Document(
                    project=self,
                    update=from_online,
                    id_=document_data['id'],
                    category=doc_category,
                    status=status,
                    **document_data,
                )
                logger.debug(f'{doc} is not available on your machine, we will download it as soon you use it.')
                n_new_documents += 1
            else:
                doc = local_docs_dict[document_data['id']]
                if 'status' in document_data:
                    status = document_data['status']
                document_data.pop('status', None)
                doc.update_meta_data(
                    category=doc_category, status=status, **document_data
                )  # reset any Document level meta data changes
                logger.debug(f'Unchanged local version of {doc} from {new_date}.')
                n_unchanged_documents += 1
            # else:
            #    logger.debug(f"Not loading Document {[document_data['id']]} with status {document_data['status'][0]}.")

        to_delete_ids = set(local_docs_dict.keys()) - updated_docs_ids_set
        n_deleted_documents = len(to_delete_ids)
        for to_del_id in to_delete_ids:
            local_docs_dict[to_del_id].delete(delete_online=False)

        logger.info(
            f'{n_updated_documents} Documents were updated,'
            f' {n_new_documents} Documents are new,'
            f' {n_unchanged_documents} Documents are unchanged,'
            f' and {n_deleted_documents} Documents were deleted.'
        )

    def get_document_by_id(self, document_id: int) -> Document:
        """Return Document by its ID."""
        for document in self._documents:
            if document.id_ == document_id:
                return document
        raise IndexError(f'Document id {document_id} was not found in {self}.')

    def del_document_by_id(self, document_id: int, delete_online: bool = False) -> Document:
        """Delete Document by its ID."""
        document = self.get_document_by_id(document_id)

        if delete_online:
            delete_file_konfuzio_api(document_id, session=self.session)
            self.write_meta_of_files()
            self.get_meta(reload=True)
            try:
                shutil.rmtree(document.document_folder)
            except FileNotFoundError:
                pass

        self._documents.remove(document)

        return None

    def get_label_by_name(self, name: str) -> Label:
        """Return Label by its name."""
        for label in self.labels:
            if label.name == name:
                return label
        raise IndexError(f'Label name {name} was not found in {self}.')

    def get_label_by_id(self, id_: int) -> Label:
        """
        Return a Label by ID.

        :param id_: ID of the Label to get.
        """
        for label in self.labels:
            if label.id_ == id_:
                return label
        raise IndexError(f'Label id {id_} was not found in {self}.')

    def get_label_set_by_name(self, name: str) -> LabelSet:
        """Return a Label Set by ID."""
        for label_set in self.label_sets:
            if label_set.name == name:
                return label_set
        raise IndexError(f'LabelSet name {name} was not found in {self}.')

    def get_label_set_by_id(self, id_: int) -> LabelSet:
        """
        Return a Label Set by ID.

        :param id_: ID of the Label Set to get.
        """
        for label_set in self.label_sets:
            if label_set.id_ == id_:
                return label_set
        raise IndexError(f'LabelSet id {id_} was not found in {self}.')

    def get_category_by_id(self, id_: int) -> Category:
        """
        Return a Category by ID.

        :param id_: ID of the Category to get.
        """
        for category in self.categories:
            if category.id_ == id_:
                return category
        raise IndexError(f'Category id {id_} was not found in {self}.')

    def get_credentials(self, key):
        """
        Return the value of the key in the credentials dict or in the config file.

        Returns None and emits a warning if the key is not found.

        :param key: Key of the credential to get.
        """
        if key in self.credentials:
            value = self.credentials[key]
        else:
            from .settings_importer import config

            value = config(key, default=None)
        if not value:
            logger.warning(f'No value found for {key} in {self}.')
        return value

    def delete(self):
        """Delete the Project folder."""
        shutil.rmtree(self.project_folder)

    def lose_weight(self):
        """Delete data of the instance."""
        super().lose_weight()
        for category in self.categories:
            category.lose_weight()
        for label_set in self.label_sets:
            label_set.lose_weight()
        for label in self.labels:
            label.lose_weight()
        self._documents = []
        self._meta_data = []
        return self

    def download_training_and_test_data(self) -> None:
        """
        Migrate your Project to another HOST.

        See https://dev.konfuzio.com/web/migration-between-konfuzio-server-instances/index.html

        """
        if len(self.documents + self.test_documents) == 0:
            raise ValueError('No Documents in the training or test set. Please add them.')
        for document in tqdm(self.documents + self.test_documents):
            document.download_document_details()
            document.get_file()
            document.get_file(ocr_version=False)
            document.get_bbox()
            document.get_images()

        print('[SUCCESS] Data exporting finished successfully!')

    def export_project_data(self, include_ais=False, training_and_test_documents=True) -> None:
        """
        "Export the Project data including Training, Test Documents and AI models.

        :include_ais: Whether to include AI models in the export
        :training_and_test_documents: Whether to include training & test documents in the export.
        """
        if training_and_test_documents:
            try:
                print('[INFO] Starting Training and Test Document export!')
                self.download_training_and_test_data()
            except Exception as error:
                print('[ERROR] Something went wrong while downloading Document data!')
                raise error
        if include_ais:
            try:
                print('[INFO] Starting AI Model file export!')
                exported_ais = export_ai_models(self)
                if exported_ais:
                    print(f'[INFO] Export finished. {exported_ais} AIs were available for export.')
                else:
                    print('[INFO] No AIs available for export.')
            except Exception as error:
                print('[ERROR] Something went wrong while downloading AIs or AI metadata!')
                raise error
