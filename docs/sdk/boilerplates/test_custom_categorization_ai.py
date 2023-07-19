"""Test code examples for the custom Categorization AI's tutorial."""
import pytest

from konfuzio_sdk.settings_importer import is_dependency_installed


@pytest.mark.skipif(
    not is_dependency_installed('timm')
    and not is_dependency_installed('torch')
    and not is_dependency_installed('transformers')
    and not is_dependency_installed('torchvision'),
    reason='Required dependencies not installed.',
)
def test_custom_categorization_ai():
    """Test creating and using a custom Categorization AI."""
    # start init
    import re

    from konfuzio_sdk.data import Document, Span, AnnotationSet, Annotation, Label
    from konfuzio_sdk.trainer.information_extraction import AbstractExtractionAI

    class CustomExtractionAI(AbstractExtractionAI):
        def extract(self, document: Document) -> Document:
            if document.annotations(use_correct=False):  # check if any Annotations exists
                return document
            else:
                # define a Label Set that will contain Labels for Annotations your Extraction AI extracts
                label_set = document.project.get_label_set_by_id(id_=document.category.id_)
                # get or create a Label that will be used for annotating
                label_name = 'Date'
                if label_name in [label.name for label in document.project.labels]:
                    label = document.project.get_label_by_name(label_name)
                else:
                    label = Label(text=label_name, project=project, label_sets=[label_set])
                for re_match in re.finditer(r'(\d+/\d+/\d+)', document.text, flags=re.MULTILINE):
                    span = Span(start_offset=re_match.span(1)[0], end_offset=re_match.span(1)[1])
                    # create Annotation Set for the Annotation. Note that every Annotation Set
                    # has to contain at least one Annotation, and Annotation always should be
                    # a part of an Annotation Set.
                    annotation_set = AnnotationSet(document=document, label_set=label_set)
                    _ = Annotation(
                        document=document,
                        label=label,
                        annotation_set=annotation_set,
                        label_set=label_set,
                        confidence=1.0,  # note that only the Annotations with confidence higher
                        # than 10% will be shown in the extracted Document.
                        spans=[span],
                    )
            return document

    # end init
    from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID

    YOUR_PROJECT_ID = TEST_PROJECT_ID
    YOUR_DOCUMENT_ID = TEST_DOCUMENT_ID
    # start usage
    import os
    from konfuzio_sdk.data import Project
    from konfuzio_sdk.trainer.document_categorization import (
        CategorizationAI,
        EfficientNet,
        PageImageCategorizationModel,
    )

    # Initialize Project and provide the AI training and test data
    project = Project(id_=YOUR_PROJECT_ID)  # see https://dev.konfuzio.com/sdk/get_started.html#example-usage

    categorization_pipeline = CategorizationAI(project.categories)
    categorization_pipeline.categories = project.categories
    categorization_pipeline.documents = [
        document for category in categorization_pipeline.categories for document in category.documents()
    ]
    categorization_pipeline.test_documents = [
        document for category in categorization_pipeline.categories for document in category.test_documents()
    ]
    # end usage
    categorization_pipeline.documents = categorization_pipeline.documents[:5]
    categorization_pipeline.test_documents = categorization_pipeline.test_documents[:5]
    # start fit
    # initialize all necessary parts of the AI – in the example we run an AI that uses images and does not use text
    categorization_pipeline.category_vocab = categorization_pipeline.build_template_category_vocab()
    # image processing model
    image_model = EfficientNet(name='efficientnet_b0')
    # building a classifier for the page images
    categorization_pipeline.classifier = PageImageCategorizationModel(
        image_model=image_model,
        output_dim=len(categorization_pipeline.category_vocab),
    )
    categorization_pipeline.build_preprocessing_pipeline(use_image=True)
    # fit the AI
    categorization_pipeline.fit(n_epochs=1, optimizer={'name': 'Adam'})

    # evaluate the AI
    data_quality = categorization_pipeline.evaluate(use_training_docs=True)
    ai_quality = categorization_pipeline.evaluate(use_training_docs=False)

    # Categorize a Document
    document = project.get_document_by_id(YOUR_DOCUMENT_ID)
    categorization_result = categorization_pipeline.categorize(document=document)
    for page in categorization_result.pages():
        print(f"Found category {page.category} for {page}")
    print(f"Found category {categorization_result.category} for {categorization_result}")

    # Save and load a pickle file for the model
    pickle_model_path = categorization_pipeline.save(reduce_weight=False)
    categorization_pipeline_loaded = CategorizationAI.load_model(pickle_model_path)
    # end fit
    assert 63 in data_quality.category_ids
    assert 63 in ai_quality.category_ids
    assert isinstance(categorization_pipeline_loaded, CategorizationAI)
    # start upload
    # from konfuzio_sdk.api import upload_ai_model, delete_ai_model
    #
    # # upload a saved model to the server
    # model_id = upload_ai_model(pickle_model_path)
    #
    # # remove model
    # delete_ai_model(model_id, ai_type='categorization')
    # end upload
    os.remove(pickle_model_path)
