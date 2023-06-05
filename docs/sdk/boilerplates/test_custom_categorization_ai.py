"""Test code examples for the custom Categorization AI's tutorial."""


def test_custom_categorization_ai():
    """Test creating and using a custom Categorization AI."""
    # start init
    from konfuzio_sdk.trainer.document_categorization import AbstractCategorizationAI
    from konfuzio_sdk.data import Page

    class CustomCategorizationAI(AbstractCategorizationAI):
        def __init__(self, *args, **kwargs):
            pass

        # initialize key variables required by the custom AI

        def fit(self):
            pass

        # Define architecture and training that the model undergoes, i.e. a NN architecture or a custom hardcoded logic
        # This method is allowed to be implemented as a no-op if you provide the trained model in other ways

        def _categorize_page(self, page: Page) -> Page:
            pass

        # define how the model assigns a Category to a Page
        # **NB:** The result of extraction must be the input Page with added Categorization attribute `Page.category`

        def save(self, path: str):
            pass

        # define how to save a model in a .pt format – for example, in a way it's defined in the CategorizationAI

    # end init
    from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID

    YOUR_PROJECT_ID = TEST_PROJECT_ID
    YOUR_DOCUMENT_ID = TEST_DOCUMENT_ID
    # start usage
    import os
    from konfuzio_sdk.data import Project
    from konfuzio_sdk.trainer.information_extraction import load_model
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
    pickle_model_path = categorization_pipeline.save(
        path=project.model_folder + '/your_model_name.pt', reduce_weight=False
    )
    categorization_pipeline_loaded = load_model(pickle_model_path)
    # end fit
    os.remove(pickle_model_path)
    assert 63 in data_quality.category_ids
    assert 63 in ai_quality.category_ids
    assert isinstance(categorization_pipeline_loaded, CategorizationAI)
