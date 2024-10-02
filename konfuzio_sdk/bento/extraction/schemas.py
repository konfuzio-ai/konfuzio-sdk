"""Define pydantic models for request and response from the Extraction AI."""
from typing import Dict, List, Optional

from pydantic import BaseModel, RootModel
from konfuzio_sdk.data import Document, Page, Project, Annotation, AnnotationSet, Span

# Use relative or top module import based on whether this is run as an actual service or imported
try:
    from ..base.utils import HexBytes
    from ..base.schemas import PageModel20240117, BboxModel20240729, AnnotationSetModel20240117, DocumentModel20240117, \
        SpanModel20240117, LabelModel20240117, AnnotationModel20240117
except (ImportError, ValueError):
    from base.utils import HexBytes
    from base.schemas import PageModel20240117, BboxModel20240729, AnnotationSetModel20240117, DocumentModel20240117, \
        SpanModel20240117, LabelModel20240117, AnnotationModel20240117


class ExtractRequest20240117(DocumentModel20240117):
    """Describe a scheme for the extraction request on 17/01/2024."""

    pass


class ExtractResponse20240117(BaseModel):
    """Describe a scheme for the extraction response on 17/01/2024."""

    annotation_sets: List[AnnotationSetModel20240117]

    @classmethod
    def process_response(cls, result) -> BaseModel:
        """
        Process a raw response from the runner to contain only selected fields.

        :param result: A raw response to be processed.
        :param schema: A schema of the response.
        :returns: A list of dictionaries with Label Set IDs and Annotation data.
        """
        annotations_result = []
        for annotation_set in result.annotation_sets():
            if not annotation_set.label_set.id_:
                continue
            current_annotation_set = {'label_set_id': annotation_set.label_set.id_, 'annotations': []}
            for annotation in annotation_set.annotations(use_correct=False, ignore_below_threshold=True):
                spans_list_of_dicts = [
                    SpanModel20240117(
                        x0=span.bbox().x0,
                        x1=span.bbox().x1,
                        y0=span.bbox().y0,
                        y1=span.bbox().y1,
                        page_index=span.page.index,
                        start_offset=span.start_offset,
                        end_offset=span.end_offset,
                        offset_string=span.offset_string,
                        offset_string_original=span.offset_string,
                    )
                    for span in annotation.spans
                ]
                current_annotation_set['annotations'].append(
                    AnnotationModel20240117(
                        offset_string=annotation.offset_string,
                        translated_string=annotation.translated_string,
                        normalized=annotation.normalized,
                        label=LabelModel20240117(
                            id=annotation.label.id_,
                            name=annotation.label.name,
                            has_multiple_top_candidates=annotation.label.has_multiple_top_candidates,
                            data_type=annotation.label.data_type,
                            threshold=annotation.label.threshold,
                        ),
                        confidence=annotation.confidence,
                        annotation_set=annotation.annotation_set.id_,
                        span=spans_list_of_dicts,
                        selection_bbox=annotation.selection_bbox,
                    )
                )
            annotations_result.append(current_annotation_set)
        return cls(annotation_sets=annotations_result)

    @classmethod
    def convert_response_to_annotations(
            cls, response: 'Response', document: Document, mappings: Optional[dict] = None
    ) -> Document:
        """
        Receive an ExtractResponse and convert it into a list of Annotations to be added to the Document.

        :param response: A Response instance to be converted.
        :param response_schema_class: The schema class.
        :param document: A Document to which the annotations should be added.
        :param mappings: A dict with "label_sets" and "labels" keys, both containing mappings from old to new IDs. Original
            IDs are used if no mapping is provided or if the mapping is not found.
        :returns: The original Document with added Annotations.
        """
        if mappings is None:
            mappings = {}

        # Mappings might be from JSON, so we need to convert keys to integers.
        label_set_mappings = {int(k): v for k, v in mappings.get('label_sets', {}).items()}
        label_mappings = {int(k): v for k, v in mappings.get('labels', {}).items()}

        response = cls(annotation_sets=response.json()['annotation_sets'])

        for annotation_set in response.annotation_sets:
            label_set_id = label_set_mappings.get(annotation_set.label_set_id, annotation_set.label_set_id)
            sdk_annotation_set = AnnotationSet(
                document=document, label_set=document.project.get_label_set_by_id(label_set_id)
            )
            for annotation in annotation_set.annotations:
                label_id = label_mappings.get(annotation.label.id, annotation.label.id)
                Annotation(
                    document=document,
                    annotation_set=sdk_annotation_set,
                    label=document.project.get_label_by_id(label_id),
                    offset_string=annotation.offset_string,
                    translated_string=annotation.translated_string,
                    normalized=annotation.normalized,
                    confidence=annotation.confidence,
                    spans=[
                        Span(
                            start_offset=span.start_offset,
                            end_offset=span.end_offset,
                        )
                        for span in annotation.span
                    ],
                )
        return document


class ExtractResponseForLegacyTrainer20240912(RootModel):
    """Describe a scheme for the legacy trainer package."""

    root: Dict

    @classmethod
    def process_response(cls, result) -> BaseModel:
        """
        Process a raw response from the runner to contain only selected fields.

        :param result: A raw response to be processed.
        :param schema: A schema of the response.
        :returns: A list of dictionaries with Label Set IDs and Annotation data.
        """
        import json
        class JSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'to_json'):
                    return obj.to_json()
                return json.JSONEncoder.default(self, obj)

        json_result = json.loads(json.dumps(result, cls=JSONEncoder))
        return cls(json_result)


    @classmethod
    def convert_response_to_annotations(
            cls, response: 'Response', document: Document, mappings: Optional[dict] = None
    ) -> Document:
        """
        Receive an ExtractResponse and convert it into a list of Annotations to be added to the Document.

        :param response: A Response instance to be converted.
        :param response_schema_class: The schema class.
        :param document: A Document to which the annotations should be added.
        :param mappings: A dict with "label_sets" and "labels" keys, both containing mappings from old to new IDs. Original
            IDs are used if no mapping is provided or if the mapping is not found.
        :returns: The original Document with added Annotations.
        """
        if mappings is None:
            mappings = {}

        # Restore Pandas Dataframe which has been converted to JSON for sending via API.
        import json
        result = response.json()

        def my_convert(res_dict):
            import pandas as pd
            new_dict = {}
            for k, v in res_dict.items():
                if isinstance(v, dict):
                    new_dict[k] = my_convert(v)
                if isinstance(v, list):
                    new_dict[k] = [my_convert(x) for x in v]
                if isinstance(v, str):
                    new_dict[k] = pd.DataFrame.from_dict(json.loads(v))
            return new_dict

        my_converted_json = my_convert(result)
        return my_converted_json
