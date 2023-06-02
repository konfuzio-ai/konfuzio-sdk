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

        def save(self):
            pass

        # define how to save a model in a .pt format – for example, in a way it's defined in the CategorizationAI

    # end init
    # from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID
    # from konfuzio_sdk.trainer.document_categorization import CategorizationAI
    #
    # YOUR_PROJECT_ID = TEST_PROJECT_ID
    # YOUR_DOCUMENT_ID = TEST_DOCUMENT_ID
    # # start example
    # from konfuzio_sdk.data import Project
    # from konfuzio_sdk.trainer.information_extraction import load_model
    #
    # # Initialize Project and provide the AI training and test data
    # project = Project(id_=YOUR_PROJECT_ID)  # see https://dev.konfuzio.com/sdk/get_started.html#example-usage
    #
    # categorization_pipeline = CustomCategorizationAI()
    # categorization_pipeline = CategorizationAI(project.categories)
    # categorization_pipeline.categories = project.categories
    # categorization_pipeline.documents = [category.documents for category in categorization_pipeline.categories]
    # categorization_pipeline.test_documents = [
    #     category.test_documents() for category in categorization_pipeline.categories
    # ]
    #
    # categorization_pipeline.documents = categorization_pipeline.documents[:5]
    # categorization_pipeline.test_documents = categorization_pipeline.test_documents[:5]
    # # Calculate features and train the AI
    # categorization_pipeline.fit()
    #
    # # Evaluate the AI
    # data_quality = categorization_pipeline.evaluate(use_training_docs=True)
    # ai_quality = categorization_pipeline.evaluate(use_training_docs=False)
    #
    # # Categorize a Document
    # document = project.get_document_by_id(YOUR_DOCUMENT_ID)
    # categorization_result = categorization_pipeline.categorize(document=document)
    # for page in categorization_result.pages():
    #     print(f"Found category {page.category} for {page}")
    # print(f"Found category {categorization_result.category} for {categorization_result}")
    #
    # # Save and load a pickle file for the model
    # pickle_model_path = categorization_pipeline.save(output_dir=project.model_folder, include_konfuzio=True)
    # categorization_pipeline_loaded = load_model(pickle_model_path)
