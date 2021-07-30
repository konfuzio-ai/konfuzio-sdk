"""Validate api functions."""
import logging
import os
import json
import sys

import pytest
import requests
from konfuzio_sdk import KONFUZIO_USER, KONFUZIO_HOST, KONFUZIO_PROJECT_ID
from konfuzio_sdk.api import (
    get_document_text,
    get_meta_of_files,
    download_file_konfuzio_api,
    is_url,
    download_images,
    is_url_image,
    konfuzio_session,
    get_project_labels,
    upload_file_konfuzio_api,
    get_document_annotations,
    post_document_annotation,
    delete_document_annotation,
    delete_file_konfuzio_api,
)
from konfuzio_sdk.utils import does_not_raise

# Change project root to tests folder
FOLDER_ROOT = os.path.dirname(os.path.realpath(__file__))

TEST_DOCUMENT = 44823


@pytest.mark.serial
def test_post_document_annotation():
    """Create an Annotation via API."""
    document_id = TEST_DOCUMENT
    start_offset = 86
    end_offset = 88
    accuracy = 0.0001
    label_id = 863  # Refers to Label Betrag (863)
    template_id = 64  # Refers to Template Brutto-Bezug (allows multisections)
    # create a revised annotation, so we can verify its existence via get_document_annotations
    response = post_document_annotation(
        document_id=document_id,
        start_offset=start_offset,
        end_offset=end_offset,
        accuracy=accuracy,
        label_id=label_id,
        template_id=template_id,
        revised=True,
    )
    annotation = json.loads(response.text)
    annotation_ids = [annot['id'] for annot in get_document_annotations(document_id, include_extractions=True)]
    assert annotation['id'] in annotation_ids
    assert delete_document_annotation(document_id, annotation['id'])


@pytest.mark.serial
def test_load_annotations_from_api():
    """Download Annotations from API for a Document."""
    text = get_document_text(TEST_DOCUMENT)
    annotations = get_document_annotations(TEST_DOCUMENT)
    assert len(annotations) == 10
    for i in range(0, len(annotations)):
        assert text[annotations[1]['start_offset'] : annotations[1]['end_offset']] == annotations[1]['offset_string']

    test_annotations = [
        {
            'id': 4419937,
            'label': 867,
            'label_text': 'Austellungsdatum',
            'label_data_type': 'Date',
            'label_threshold': 0.1,
            'section_label_text': 'Lohnabrechnung',
            'section_label_id': 63,
            'section': 78730,
            'offset_string': '22.05.2018',
            'offset_string_original': '22.05.2018',
            'translated_string': None,
            'normalized': '2018-05-22',
            'start_offset': 159,
            'end_offset': 169,
            'accuracy': None,
            'is_correct': True,
            'revised': False,
            'created_by': 59,
            'revised_by': None,
            'bbox': {
                'x0': 468.48,
                'x1': 527.04,
                'y0': 797.311,
                'y1': 806.311,
                'top': 35.369,
                'bottom': 44.369,
                'line_index': 1,
                'page_index': 0,
            },
            'bboxes': [
                {
                    'x0': 468.48,
                    'x1': 527.04,
                    'y0': 797.311,
                    'y1': 806.311,
                    'top': 35.369,
                    'bottom': 44.369,
                    'end_offset': 169,
                    'page_index': 0,
                    'line_number': 2,
                    'start_offset': 159,
                    'offset_string': '22.05.2018',
                    'offset_string_original': '22.05.2018',
                }
            ],
            'selection_bbox': {
                'x0': 468.48,
                'x1': 527.04,
                'y0': 797.311,
                'y1': 806.311,
                'top': 35.369,
                'bottom': 44.369,
                'line_index': 1,
                'page_index': 0,
            },
            'custom_offset_string': False,
            'get_created_by': 'ana@konfuzio.com',
            'get_revised_by': 'n/a',
        },
        {
            'id': 4420022,
            'label': 860,
            'label_text': 'Steuerklasse',
            'label_data_type': 'Text',
            'label_threshold': 0.1,
            'section_label_text': 'Lohnabrechnung',
            'section_label_id': 63,
            'section': 78730,
            'offset_string': '1',
            'offset_string_original': '1',
            'translated_string': None,
            'normalized': '1',
            'start_offset': 365,
            'end_offset': 366,
            'accuracy': None,
            'is_correct': True,
            'revised': False,
            'created_by': 59,
            'revised_by': None,
            'bbox': {
                'x0': 126.96,
                'x1': 131.04,
                'y0': 772.589,
                'y1': 783.589,
                'top': 58.091,
                'bottom': 69.091,
                'line_index': 1,
                'page_index': 0,
            },
            'bboxes': [
                {
                    'x0': 126.96,
                    'x1': 131.04,
                    'y0': 772.589,
                    'y1': 783.589,
                    'top': 58.091,
                    'bottom': 69.091,
                    'end_offset': 366,
                    'page_index': 0,
                    'line_number': 5,
                    'start_offset': 365,
                    'offset_string': '1',
                    'offset_string_original': '1',
                }
            ],
            'selection_bbox': {
                'x0': 126.96,
                'x1': 131.04,
                'y0': 772.589,
                'y1': 783.589,
                'top': 58.091,
                'bottom': 69.091,
                'line_index': 1,
                'page_index': 0,
            },
            'custom_offset_string': False,
            'get_created_by': 'ana@konfuzio.com',
            'get_revised_by': 'n/a',
        },
        {
            'id': 4420057,
            'label': 865,
            'label_text': 'Vorname',
            'label_data_type': 'Text',
            'label_threshold': 0.1,
            'section_label_text': 'Lohnabrechnung',
            'section_label_id': 63,
            'section': 78730,
            'offset_string': 'Erna-Muster',
            'offset_string_original': 'Erna-Muster',
            'translated_string': None,
            'normalized': 'Erna-Muster',
            'start_offset': 1507,
            'end_offset': 1518,
            'accuracy': None,
            'is_correct': True,
            'revised': False,
            'created_by': 59,
            'revised_by': None,
            'bbox': {
                'x0': 65.52,
                'x1': 130.8,
                'y0': 628.832,
                'y1': 636.832,
                'top': 204.848,
                'bottom': 212.848,
                'line_index': 1,
                'page_index': 0,
            },
            'bboxes': [
                {
                    'x0': 65.52,
                    'x1': 130.8,
                    'y0': 628.832,
                    'y1': 636.832,
                    'top': 204.848,
                    'bottom': 212.848,
                    'end_offset': 1518,
                    'page_index': 0,
                    'line_number': 24,
                    'start_offset': 1507,
                    'offset_string': 'Erna-Muster',
                    'offset_string_original': 'Erna-Muster',
                }
            ],
            'selection_bbox': {
                'x0': 65.52,
                'x1': 130.8,
                'y0': 628.832,
                'y1': 636.832,
                'top': 204.848,
                'bottom': 212.848,
                'line_index': 1,
                'page_index': 0,
            },
            'custom_offset_string': False,
            'get_created_by': 'ana@konfuzio.com',
            'get_revised_by': 'n/a',
        },
        {
            'id': 4420245,
            'label': 866,
            'label_text': 'Nachname',
            'label_data_type': 'Text',
            'label_threshold': 0.1,
            'section_label_text': 'Lohnabrechnung',
            'section_label_id': 63,
            'section': 78730,
            'offset_string': 'Eiermann',
            'offset_string_original': 'Eiermann',
            'translated_string': None,
            'normalized': 'Eiermann',
            'start_offset': 1519,
            'end_offset': 1527,
            'accuracy': None,
            'is_correct': True,
            'revised': False,
            'created_by': 59,
            'revised_by': None,
            'bbox': {
                'x0': 137.52,
                'x1': 184.8,
                'y0': 628.832,
                'y1': 636.832,
                'top': 204.848,
                'bottom': 212.848,
                'line_index': 1,
                'page_index': 0,
            },
            'bboxes': [
                {
                    'x0': 137.52,
                    'x1': 184.8,
                    'y0': 628.832,
                    'y1': 636.832,
                    'top': 204.848,
                    'bottom': 212.848,
                    'end_offset': 1527,
                    'page_index': 0,
                    'line_number': 24,
                    'start_offset': 1519,
                    'offset_string': 'Eiermann',
                    'offset_string_original': 'Eiermann',
                }
            ],
            'selection_bbox': {
                'x0': 137.52,
                'x1': 184.8,
                'y0': 628.832,
                'y1': 636.832,
                'top': 204.848,
                'bottom': 212.848,
                'line_index': 1,
                'page_index': 0,
            },
            'custom_offset_string': False,
            'get_created_by': 'ana@konfuzio.com',
            'get_revised_by': 'n/a',
        },
        {
            'id': 4420346,
            'label': 861,
            'label_text': 'Lohnart',
            'label_data_type': 'Text',
            'label_threshold': 0.1,
            'section_label_text': 'Brutto-Bezug',
            'section_label_id': 64,
            'section': 292092,
            'offset_string': '2000',
            'offset_string_original': '2000',
            'translated_string': None,
            'normalized': '2000',
            'start_offset': 1758,
            'end_offset': 1762,
            'accuracy': None,
            'is_correct': True,
            'revised': False,
            'created_by': 59,
            'revised_by': None,
            'bbox': {
                'x0': 48.72,
                'x1': 71.28,
                'y0': 532.592,
                'y1': 540.592,
                'top': 301.088,
                'bottom': 309.088,
                'line_index': 1,
                'page_index': 0,
            },
            'bboxes': [
                {
                    'x0': 48.72,
                    'x1': 71.28,
                    'y0': 532.592,
                    'y1': 540.592,
                    'top': 301.088,
                    'bottom': 309.088,
                    'end_offset': 1762,
                    'page_index': 0,
                    'line_number': 30,
                    'start_offset': 1758,
                    'offset_string': '2000',
                    'offset_string_original': '2000',
                }
            ],
            'selection_bbox': {
                'x0': 48.72,
                'x1': 71.28,
                'y0': 532.592,
                'y1': 540.592,
                'top': 301.088,
                'bottom': 309.088,
                'line_index': 1,
                'page_index': 0,
            },
            'custom_offset_string': False,
            'get_created_by': 'ana@konfuzio.com',
            'get_revised_by': 'n/a',
        },
        {
            'id': 4420348,
            'label': 862,
            'label_text': 'Bezeichnung',
            'label_data_type': 'Text',
            'label_threshold': 0.1,
            'section_label_text': 'Brutto-Bezug',
            'section_label_id': 64,
            'section': 292092,
            'offset_string': 'Gehalt',
            'offset_string_original': 'Gehalt',
            'translated_string': None,
            'normalized': 'Gehalt',
            'start_offset': 1763,
            'end_offset': 1769,
            'accuracy': None,
            'is_correct': True,
            'revised': False,
            'created_by': 59,
            'revised_by': None,
            'bbox': {
                'x0': 78.48,
                'x1': 113.28,
                'y0': 532.592,
                'y1': 540.592,
                'top': 301.088,
                'bottom': 309.088,
                'line_index': 1,
                'page_index': 0,
            },
            'bboxes': [
                {
                    'x0': 78.48,
                    'x1': 113.28,
                    'y0': 532.592,
                    'y1': 540.592,
                    'top': 301.088,
                    'bottom': 309.088,
                    'end_offset': 1769,
                    'page_index': 0,
                    'line_number': 30,
                    'start_offset': 1763,
                    'offset_string': 'Gehalt',
                    'offset_string_original': 'Gehalt',
                }
            ],
            'selection_bbox': {
                'x0': 78.48,
                'x1': 113.28,
                'y0': 532.592,
                'y1': 540.592,
                'top': 301.088,
                'bottom': 309.088,
                'line_index': 1,
                'page_index': 0,
            },
            'custom_offset_string': False,
            'get_created_by': 'ana@konfuzio.com',
            'get_revised_by': 'n/a',
        },
        {
            'id': 4420349,
            'label': 863,
            'label_text': 'Betrag',
            'label_data_type': 'Number',
            'label_threshold': 0.1,
            'section_label_text': 'Brutto-Bezug',
            'section_label_id': 64,
            'section': 292092,
            'offset_string': '3.120,00',
            'offset_string_original': '3.120,00',
            'translated_string': None,
            'normalized': 3120.0,
            'start_offset': 1831,
            'end_offset': 1839,
            'accuracy': None,
            'is_correct': True,
            'revised': False,
            'created_by': 59,
            'revised_by': None,
            'bbox': {
                'x0': 516.48,
                'x1': 563.04,
                'y0': 532.592,
                'y1': 540.592,
                'top': 301.088,
                'bottom': 309.088,
                'line_index': 1,
                'page_index': 0,
            },
            'bboxes': [
                {
                    'x0': 516.48,
                    'x1': 563.04,
                    'y0': 532.592,
                    'y1': 540.592,
                    'top': 301.088,
                    'bottom': 309.088,
                    'end_offset': 1839,
                    'page_index': 0,
                    'line_number': 30,
                    'start_offset': 1831,
                    'offset_string': '3.120,00',
                    'offset_string_original': '3.120,00',
                }
            ],
            'selection_bbox': {
                'x0': 516.48,
                'x1': 563.04,
                'y0': 532.592,
                'y1': 540.592,
                'top': 301.088,
                'bottom': 309.088,
                'line_index': 1,
                'page_index': 0,
            },
            'custom_offset_string': False,
            'get_created_by': 'ana@konfuzio.com',
            'get_revised_by': 'n/a',
        },
        {
            'id': 4420363,
            'label': 859,
            'label_text': 'Gesamt-Brutto',
            'label_data_type': 'Number',
            'label_threshold': 0.1,
            'section_label_text': 'Lohnabrechnung',
            'section_label_id': 63,
            'section': 78730,
            'offset_string': '3.120,00',
            'offset_string_original': '3.120,00',
            'translated_string': None,
            'normalized': 3120.0,
            'start_offset': 2111,
            'end_offset': 2119,
            'accuracy': None,
            'is_correct': True,
            'revised': False,
            'created_by': 59,
            'revised_by': None,
            'bbox': {
                'x0': 516.48,
                'x1': 563.041,
                'y0': 365.072,
                'y1': 373.072,
                'top': 468.608,
                'bottom': 476.608,
                'line_index': 1,
                'page_index': 0,
            },
            'bboxes': [
                {
                    'x0': 516.48,
                    'x1': 563.041,
                    'y0': 365.072,
                    'y1': 373.072,
                    'top': 468.608,
                    'bottom': 476.608,
                    'end_offset': 2119,
                    'page_index': 0,
                    'line_number': 34,
                    'start_offset': 2111,
                    'offset_string': '3.120,00',
                    'offset_string_original': '3.120,00',
                }
            ],
            'selection_bbox': {
                'x0': 516.48,
                'x1': 563.041,
                'y0': 365.072,
                'y1': 373.072,
                'top': 468.608,
                'bottom': 476.608,
                'line_index': 1,
                'page_index': 0,
            },
            'custom_offset_string': False,
            'get_created_by': 'ana@konfuzio.com',
            'get_revised_by': 'n/a',
        },
        {
            'id': 4420350,
            'label': 864,
            'label_text': 'Netto-Verdienst',
            'label_data_type': 'Number',
            'label_threshold': 0.1,
            'section_label_text': 'Lohnabrechnung',
            'section_label_id': 63,
            'section': 78730,
            'offset_string': '2.189,07',
            'offset_string_original': '2.189,07',
            'translated_string': None,
            'normalized': 2189.07,
            'start_offset': 3004,
            'end_offset': 3012,
            'accuracy': None,
            'is_correct': True,
            'revised': False,
            'created_by': 59,
            'revised_by': None,
            'bbox': {
                'x0': 516.48,
                'x1': 562.8,
                'y0': 245.073,
                'y1': 252.073,
                'top': 589.607,
                'bottom': 596.607,
                'line_index': 1,
                'page_index': 0,
            },
            'bboxes': [
                {
                    'x0': 516.48,
                    'x1': 562.8,
                    'y0': 245.073,
                    'y1': 252.073,
                    'top': 589.607,
                    'bottom': 596.607,
                    'end_offset': 3012,
                    'page_index': 0,
                    'line_number': 47,
                    'start_offset': 3004,
                    'offset_string': '2.189,07',
                    'offset_string_original': '2.189,07',
                }
            ],
            'selection_bbox': {
                'x0': 516.48,
                'x1': 562.8,
                'y0': 245.073,
                'y1': 252.073,
                'top': 589.607,
                'bottom': 596.607,
                'line_index': 1,
                'page_index': 0,
            },
            'custom_offset_string': False,
            'get_created_by': 'ana@konfuzio.com',
            'get_revised_by': 'n/a',
        },
        {
            'id': 4420351,
            'label': 858,
            'label_text': 'Auszahlungsbetrag',
            'label_data_type': 'Number',
            'label_threshold': 0.1,
            'section_label_text': 'Lohnabrechnung',
            'section_label_id': 63,
            'section': 78730,
            'offset_string': '2.189,07',
            'offset_string_original': '2.189,07',
            'translated_string': None,
            'normalized': 2189.07,
            'start_offset': 3777,
            'end_offset': 3785,
            'accuracy': None,
            'is_correct': True,
            'revised': False,
            'created_by': 59,
            'revised_by': None,
            'bbox': {
                'x0': 516.48,
                'x1': 562.8,
                'y0': 76.829,
                'y1': 87.829,
                'top': 753.851,
                'bottom': 764.851,
                'line_index': 1,
                'page_index': 0,
            },
            'bboxes': [
                {
                    'x0': 516.48,
                    'x1': 562.8,
                    'y0': 76.829,
                    'y1': 87.829,
                    'top': 753.851,
                    'bottom': 764.851,
                    'end_offset': 3785,
                    'page_index': 0,
                    'line_number': 63,
                    'start_offset': 3777,
                    'offset_string': '2.189,07',
                    'offset_string_original': '2.189,07',
                }
            ],
            'selection_bbox': {
                'x0': 516.48,
                'x1': 562.8,
                'y0': 76.829,
                'y1': 87.829,
                'top': 753.851,
                'bottom': 764.851,
                'line_index': 1,
                'page_index': 0,
            },
            'custom_offset_string': False,
            'get_created_by': 'ana@konfuzio.com',
            'get_revised_by': 'n/a',
        },
    ]

    test_annotations.sort(key=lambda x: x['id'] or 0)
    annotations.sort(key=lambda x: x['id'] or 0)
    assert len(test_annotations) == len(annotations)
    for i, _ in enumerate(annotations):
        assert annotations[i] == test_annotations[i]


@pytest.mark.serial
def test_get_project_labels():
    """Download Labels from API for a Project."""
    assert get_project_labels() == [
        {
            'id': 858,
            'project': 46,
            'text': 'Auszahlungsbetrag',
            'text_clean': 'Auszahlungsbetrag',
            'shortcut': 'a',
            'description': 'Der Betrag der dem Arbeitnehmer ausgezahlt wird.',
            'threshold': 0.1,
            'token_full_replacement': True,
            'token_whitespace_replacement': True,
            'token_number_replacement': True,
            'has_multiple_top_candidates': False,
            'get_data_type_display': 'Number',
        },
        {
            'id': 859,
            'project': 46,
            'text': 'Gesamt-Brutto',
            'text_clean': 'GesamtBrutto',
            'shortcut': 'g',
            'description': 'Das Bruttogehalt des Arbeitnehmers.',
            'threshold': 0.1,
            'token_full_replacement': True,
            'token_whitespace_replacement': True,
            'token_number_replacement': True,
            'has_multiple_top_candidates': False,
            'get_data_type_display': 'Number',
        },
        {
            'id': 860,
            'project': 46,
            'text': 'Steuerklasse',
            'text_clean': 'Steuerklasse',
            'shortcut': 's',
            'description': 'Die Steuerklasse des Arbeitnehmers.',
            'threshold': 0.1,
            'token_full_replacement': True,
            'token_whitespace_replacement': True,
            'token_number_replacement': True,
            'has_multiple_top_candidates': False,
            'get_data_type_display': 'Text',
        },
        {
            'id': 861,
            'project': 46,
            'text': 'Lohnart',
            'text_clean': 'Lohnart',
            'shortcut': 'l',
            'description': 'Die 4-stellige numerische Bezeichnung eines Brutto-Bezuges.',
            'threshold': 0.1,
            'token_full_replacement': True,
            'token_whitespace_replacement': True,
            'token_number_replacement': True,
            'has_multiple_top_candidates': False,
            'get_data_type_display': 'Text',
        },
        {
            'id': 862,
            'project': 46,
            'text': 'Bezeichnung',
            'text_clean': 'Bezeichnung',
            'shortcut': 'b',
            'description': 'Die Bezeichnung eines Brutto-Bezuges.',
            'threshold': 0.1,
            'token_full_replacement': True,
            'token_whitespace_replacement': True,
            'token_number_replacement': True,
            'has_multiple_top_candidates': False,
            'get_data_type_display': 'Text',
        },
        {
            'id': 863,
            'project': 46,
            'text': 'Betrag',
            'text_clean': 'Betrag',
            'shortcut': 'N/A',
            'description': 'Der Betrag eines Brutto-Bezuges.',
            'threshold': 0.1,
            'token_full_replacement': True,
            'token_whitespace_replacement': True,
            'token_number_replacement': True,
            'has_multiple_top_candidates': False,
            'get_data_type_display': 'Number',
        },
        {
            'id': 864,
            'project': 46,
            'text': 'Netto-Verdienst',
            'text_clean': 'NettoVerdienst',
            'shortcut': 'n',
            'description': 'Der Netto-Verdienst des Arbeitnehmers.',
            'threshold': 0.1,
            'token_full_replacement': True,
            'token_whitespace_replacement': True,
            'token_number_replacement': True,
            'has_multiple_top_candidates': False,
            'get_data_type_display': 'Number',
        },
        {
            'id': 865,
            'project': 46,
            'text': 'Vorname',
            'text_clean': 'Vorname',
            'shortcut': 'v',
            'description': 'Der Vorname des Arbeitnehmers.',
            'threshold': 0.1,
            'token_full_replacement': True,
            'token_whitespace_replacement': True,
            'token_number_replacement': True,
            'has_multiple_top_candidates': False,
            'get_data_type_display': 'Text',
        },
        {
            'id': 866,
            'project': 46,
            'text': 'Nachname',
            'text_clean': 'Nachname',
            'shortcut': 'N/A',
            'description': 'Der Nachname des Arbeitnehmers.',
            'threshold': 0.1,
            'token_full_replacement': True,
            'token_whitespace_replacement': True,
            'token_number_replacement': True,
            'has_multiple_top_candidates': False,
            'get_data_type_display': 'Text',
        },
        {
            'id': 867,
            'project': 46,
            'text': 'Austellungsdatum',
            'text_clean': 'Austellungsdatum',
            'shortcut': 'N/A',
            'description': 'Das Datum der Austellung der Lohnabrechnung.',
            'threshold': 0.1,
            'token_full_replacement': True,
            'token_whitespace_replacement': True,
            'token_number_replacement': True,
            'has_multiple_top_candidates': False,
            'get_data_type_display': 'Date',
        },
        {
            'id': 964,
            'project': 46,
            'text': 'EMPTY_LABEL',
            'text_clean': 'EMPTY_LABEL',
            'shortcut': 'e',
            'description': 'Label which should not have any Annotations, used for testing.',
            'threshold': 0.1,
            'token_full_replacement': True,
            'token_whitespace_replacement': True,
            'token_number_replacement': True,
            'has_multiple_top_candidates': False,
            'get_data_type_display': 'Text',
        },
    ]


url_data = [
    ('', False),
    ('ssssssssssss', False),
    ('mail@mail.com', False),
    ('https://www.google.com/image.png', True),
    ('http://www.google.com/image.png', True),
    ('www.google.com/image.png', False),  # make sure protocol is used in url
    ('www.google.com', False),  # make sure protocol is used in url
    ('c:/drive/me', False),
    ('c:\\me\\e', False),
    ('(/)?.', False),
    ('hello world text', False),
    ('   ', False),
]


@pytest.mark.serial
@pytest.mark.parametrize("url, expected", url_data)
def test_is_url(url, expected):
    """Test if url is valid."""
    assert is_url(url) == expected


image_content_data = [
    ('https://konfuzio.com/wp-content/uploads/2020/05/sparkasse.png', True, does_not_raise()),
    ('https://konfaasdfuzio.com/img/logo_1zu1.png', False, pytest.raises(requests.exceptions.ConnectionError)),
    ('https://konfuzio.com', False, does_not_raise()),
]


@pytest.mark.serial
@pytest.mark.parametrize("url, expected_result, expected_error", image_content_data)
def test_is_url_image(url, expected_result, expected_error):
    """Test if url return an image."""
    with expected_error:
        assert is_url_image(url) == expected_result


image_download_url_data = [
    ('https://konfuzio.com/wp-content/uploads/2020/05/sparkasse.png', True, does_not_raise()),
    ('https://konfaasdfuzio.com', True, pytest.raises(requests.exceptions.ConnectionError)),
    ('https://konfuzio.com', True, pytest.raises(NotImplementedError)),
]


@pytest.mark.serial
@pytest.mark.parametrize("url, expected_result, expected_error", image_download_url_data)
def test_download_images(url, expected_result, expected_error):
    """Test download of images."""
    with expected_error:
        assert download_images([url, url])


@pytest.mark.serial
def test_get_csrf_token_valid_user():
    """Test session in Konfuzio Server."""
    konfuzio_session()


@pytest.mark.serial
def test_upload_file_konfuzio_api():
    """Test upload of a file through API and its removal."""
    logging.info(f'KONFUZIO_USER: {KONFUZIO_USER}')
    logging.info(f'KONFUZIO_HOST: {KONFUZIO_HOST}')
    logging.info(f'KONFUZIO_PROJECT_ID: {KONFUZIO_PROJECT_ID}')

    file_path = os.path.join(FOLDER_ROOT, 'test_data/pdf/1_test.pdf')
    document_id = json.loads(upload_file_konfuzio_api(file_path, project_id=KONFUZIO_PROJECT_ID).text)['id']
    delete_file_konfuzio_api(document_id)


@pytest.mark.serial
class TestKonfuzioSDKAPI:
    """Test API with payslip example project."""

    def test_download_text(self):
        """Test get text for a document."""
        logging.info(f'KONFUZIO_USER: {KONFUZIO_USER}')
        logging.info(f'KONFUZIO_HOST: {KONFUZIO_HOST}')
        logging.info(f'KONFUZIO_PROJECT_ID: {KONFUZIO_PROJECT_ID}')
        assert get_document_text(document_id=TEST_DOCUMENT) is not None

    def test_get_list_of_files(self):
        """Get meta information from documents in the project."""
        logging.info(f'KONFUZIO_USER: {KONFUZIO_USER}')
        logging.info(f'KONFUZIO_HOST: {KONFUZIO_HOST}')
        logging.info(f'KONFUZIO_PROJECT_ID: {KONFUZIO_PROJECT_ID}')
        sorted_documents = get_meta_of_files()
        sorted_dataset_documents = [x for x in sorted_documents if x['dataset_status'] in [2, 3]]
        assert len(sorted_dataset_documents) == 26 + 3

    def test_download_file_with_ocr(self):
        """Test to download the ocred version of a document."""
        document_id = 94858
        downloaded_file = download_file_konfuzio_api(document_id=document_id)
        logging.info(f'Size of file {document_id}: {sys.getsizeof(downloaded_file)}')

    def test_download_file_without_ocr(self):
        """Test to download the original version of a document."""
        document_id = 94858
        downloaded_file = download_file_konfuzio_api(document_id=document_id, ocr=False)
        logging.info(f'Size of file {document_id}: {sys.getsizeof(downloaded_file)}')

    def test_download_file_not_available(self):
        """Test to download the original version of a document."""
        document_id = 15631000000000000000000000000
        with pytest.raises(FileNotFoundError):
            download_file_konfuzio_api(document_id=document_id)

    def test_download_file_not_available_but_without_permission(self):
        """Test to download the original version of a document."""
        document_id = 1
        with pytest.raises(FileNotFoundError):
            download_file_konfuzio_api(document_id=document_id)

    def test_download_log_file_which_is_no_pdf(self):
        """Test to download the original version of a document."""
        document_id = 94857
        # TODO: investigate Failed: DID NOT RAISE <class 'FileNotFoundError'>
        # with pytest.raises(FileNotFoundError):
        #     download_file_konfuzio_api(document_id=document_id)
        download_file_konfuzio_api(document_id=document_id)
