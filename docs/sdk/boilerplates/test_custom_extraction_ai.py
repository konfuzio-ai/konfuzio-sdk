"""Test code examples for the creation of a custom Extraction AI tutorial."""


def test_create_extraction_ai():
    """Test creating a custom Extraction AI."""
    # start custom
    from konfuzio_sdk.trainer.information_extraction import AbstractExtractionAI
    from konfuzio_sdk.data import Document

    class CustomExtractionAI(AbstractExtractionAI):
        def __init__(self, *args, **kwargs):
            pass

        # initialize key variables required by the custom AI

        def fit(self):
            pass

        # Define architecture and training that the model undergoes, i.e. a NN architecture or a custom hardcoded logic
        # This method is allowed to be implemented as a no-op if you provide the trained model in other ways

        def extract(self, document: Document) -> Document:
            pass

        # define how the model extracts information from Documents
        # **NB:** The result of extraction must be a copy of the input Document with added Annotations attribute
        # `Document._annotations`

        def check_is_ready(self) -> bool:
            pass

        # define if all components needed for training/prediction are set

    # end custom

    ai = CustomExtractionAI()
    assert isinstance(ai, CustomExtractionAI)

    # from tests.variables import TEST_PROJECT_ID
    #
    # YOUR_PROJECT_ID = TEST_PROJECT_ID
    # from konfuzio_sdk.data import Project, Document
    # from konfuzio_sdk.trainer.information_extraction import load_model

    # Initialize Project and provide the AI training and test data
    # project = Project(id_=YOUR_PROJECT_ID)  # see https://dev.konfuzio.com/sdk/get_started.html#example-usage
    #
    # extraction_pipeline = CustomExtractionAI()
    # extraction_pipeline.category = project.get_category_by_id(id_=YOUR_CATEGORY_ID)
    # extraction_pipeline.documents = extraction_pipeline.category.documents()
    # extraction_pipeline.test_documents = extraction_pipeline.category.test_documents()
    #
    # # Train the AI
    # extraction_pipeline.fit()
    #
    # # Evaluate the AI
    # data_quality = extraction_pipeline.evaluate_full(use_training_docs=True)
    # ai_quality = extraction_pipeline.evaluate_full(use_training_docs=False)
    #
    # # Extract a Document
    # document = self.project.get_document_by_id(YOUR_DOCUMENT_ID)
    # extraction_result: Document = extraction_pipeline.extract(document=document)
    #
    # # Save and load a pickle file for the model
    # pickle_model_path = extraction_pipeline.save(output_dir=project.model_folder, include_konfuzio=True)
    # extraction_pipeline_loaded = load_model(pickle_model_path)
