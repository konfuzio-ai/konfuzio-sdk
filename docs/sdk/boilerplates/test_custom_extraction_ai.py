"""Test code examples for the creation of a custom Extraction AI tutorial."""


def test_create_extraction_ai():
    """Test creating a custom Extraction AI."""
    # start custom
    import re

    from konfuzio_sdk.data import Document, Span, Annotation, Label
    from konfuzio_sdk.trainer.information_extraction import AbstractExtractionAI

    class CustomExtractionAI(AbstractExtractionAI):
        def extract(self, document: Document) -> Document:
            """Extract regex matches for dates."""
            # call the parent method to get a Virtual Document with no Annotations and with the Category changed to the
            # one saved within the Extraction AI
            document = super().extract(document)

            # define a Label Set that will contain Labels for Annotations your Extraction AI extracts
            # here we use the default Label Set of the Category
            label_set = document.category.get_default_label_set()
            # get or create a Label that will be used for annotating
            label_name = 'Date'
            if label_name in [label.name for label in document.category.labels]:
                label = document.project.get_label_by_name(label_name)
            else:
                label = Label(text=label_name, project=project, label_sets=[label_set])
            annotation_set = document.get_default_annotation_set()
            for re_match in re.finditer(r'(\d+/\d+/\d+)', document.text, flags=re.MULTILINE):
                span = Span(start_offset=re_match.span(1)[0], end_offset=re_match.span(1)[1])
                # create Annotation Set for the Annotation. Note that every Annotation Set
                # has to contain at least one Annotation, and Annotation always should be
                # a part of an Annotation Set.
                _ = Annotation(
                    document=document,
                    label=label,
                    annotation_set=annotation_set,
                    confidence=1.0,  # note that by default, only the Annotations with confidence higher than 10%
                    # will be shown in the extracted Document. This can be changed in the Label settings UI.
                    spans=[span],
                )
            return document

    # end custom
    from tests.variables import TEST_PROJECT_ID, TEST_PAYSLIPS_CATEGORY_ID

    YOUR_PROJECT_ID = TEST_PROJECT_ID
    YOUR_CATEGORY_ID = TEST_PAYSLIPS_CATEGORY_ID
    # start init_ai
    import os
    from konfuzio_sdk.data import Project

    # Initialize Project and provide the AI training and test data
    project = Project(id_=YOUR_PROJECT_ID)  # see https://dev.konfuzio.com/sdk/get_started.html#example-usage
    category = project.get_category_by_id(YOUR_CATEGORY_ID)
    categorization_pipeline = CustomExtractionAI(category)

    # Create a sample test Document to run extraction on
    example_text = """
        19/05/1996 is my birthday.
        04/07/1776 is the Independence day.
        """
    sample_document = Document(project=project, text=example_text, category=category)

    # Extract a Document
    extracted = categorization_pipeline.extract(sample_document)
    # we set use_correct=False because we didn't change the default flag is_correct=False upon creating the Annotations
    assert len(extracted.annotations(use_correct=False)) == 2

    # Save and load a pickle file for the model
    pickle_model_path = categorization_pipeline.save()
    extraction_pipeline_loaded = CustomExtractionAI.load_model(pickle_model_path)
    # end init_ai
    extraction_pipeline_loaded
    os.remove(pickle_model_path)
