"""Test code examples for the creation of a custom Extraction AI tutorial."""


def test_create_extraction_ai():
    """Test creating a custom Extraction AI."""
    # start custom
    from konfuzio_sdk.trainer.information_extraction import AbstractExtractionAI
    from konfuzio_sdk.data import Document, Category

    class CustomExtractionAI(AbstractExtractionAI):
        def __init__(self, category: Category, *args, **kwargs):
            super().__init__(category)
            pass

        # initialize key variables required by the custom AI
        # for instance, self.category to define within which Category the Extraction takes place

        def fit(self):
            pass

        # Define architecture and training that the model undergoes, i.e. a NN architecture or a custom hardcoded logic
        # for instance:
        #
        # self.clf = RandomForestClassifier(
        #             class_weight="balanced", random_state=100
        #         )
        # self.clf.fit(self.df_train[self.label_feature_list], self.df_train['target'])
        #
        # This method is allowed to be implemented as a no-op if you provide the trained model in other ways

        def extract(self, document: Document) -> Document:
            pass

        # define how the model extracts information from Documents
        # for instance:
        #
        # inference_document = deepcopy(document)
        # self.tokenizer.tokenize(inference_document)
        # df, _feature_names, _raw_errors = self.features(inference_document)
        # extracted = self.extract_from_df(df, inference_document)
        #
        # **NB:** The result of extraction must be a copy of the input Document with added Annotations attribute
        # `Document._annotations`

        def check_is_ready(self) -> bool:
            pass

        # define if all components needed for training/prediction are set, for instance, is self.tokenizer set or does
        # self.category contain training and testing Documents.

    # end custom

    from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID

    YOUR_PROJECT_ID = TEST_PROJECT_ID
    YOUR_CATEGORY_ID = 63
    YOUR_DOCUMENT_ID = TEST_DOCUMENT_ID
    from konfuzio_sdk.trainer.information_extraction import RFExtractionAI
    from konfuzio_sdk.tokenizer.base import ListTokenizer

    # start init_ai
    from konfuzio_sdk.data import Project, Document
    from konfuzio_sdk.trainer.information_extraction import load_model

    # Initialize Project and the AI
    project = Project(id_=YOUR_PROJECT_ID)  # see https://dev.konfuzio.com/sdk/get_started.html#example-usage
    category = project.get_category_by_id(YOUR_CATEGORY_ID)
    extraction_pipeline = CustomExtractionAI(category)
    # end init_ai
    assert isinstance(extraction_pipeline, CustomExtractionAI)
    project = Project(id_=YOUR_PROJECT_ID, strict_data_validation=False)
    extraction_pipeline = RFExtractionAI()
    extraction_pipeline.tokenizer = ListTokenizer(tokenizers=[])
    # start category
    # provide the categories, training and test data
    extraction_pipeline.category = category
    extraction_pipeline.documents = extraction_pipeline.category.documents()
    extraction_pipeline.test_documents = extraction_pipeline.category.test_documents()
    # end category
    extraction_pipeline.documents = extraction_pipeline.documents[:5]
    extraction_pipeline.df_train, extraction_pipeline.label_feature_list = extraction_pipeline.feature_function(
        documents=extraction_pipeline.documents, require_revised_annotations=False
    )
    # start train
    # Train the AI
    extraction_pipeline.fit()

    # Evaluate the AI
    data_quality = extraction_pipeline.evaluate_full(use_training_docs=True)
    ai_quality = extraction_pipeline.evaluate_full(use_training_docs=False)

    # Extract a Document
    document = project.get_document_by_id(YOUR_DOCUMENT_ID)
    extraction_result: Document = extraction_pipeline.extract(document=document)

    # Save and load a pickle file for the model
    pickle_model_path = extraction_pipeline.save(output_dir=project.model_folder, include_konfuzio=True)
    extraction_pipeline_loaded = load_model(pickle_model_path)
    # end train
    from konfuzio_sdk.evaluate import ExtractionEvaluation

    assert isinstance(data_quality, ExtractionEvaluation)
    assert isinstance(ai_quality, ExtractionEvaluation)
    assert isinstance(extraction_result, Document)
    assert isinstance(extraction_pipeline_loaded, RFExtractionAI)
