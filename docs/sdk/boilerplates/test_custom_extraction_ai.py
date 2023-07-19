"""Test code examples for the creation of a custom Extraction AI tutorial."""


def test_create_extraction_ai():
    """Test creating a custom Extraction AI."""
    # start custom
    from konfuzio_sdk.trainer.information_extraction import AbstractExtractionAI
    from konfuzio_sdk.data import Document, Category

    class CustomExtractionAI(AbstractExtractionAI):
        def __init__(self, category: Category, *args, **kwargs):
            # we need to specify Category because it defines which Labels will be used in the Extraction, i.e. which
            # Labels will be present in the processed Document.
            #
            # the Labels can be created manually or dynamically – during extraction process – the latter is possible
            # when "Create Labels and Label Sets" is enabled in a Superuser Project's settings, if a user is a Supeuser
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
        # This method does not return anything; rather, it modifies the self.model if you provide this attribute.
        #
        # This method is allowed to be implemented as a no-op if you provide the trained model in other ways

        def extract(self, document: Document) -> Document:
            pass

        # Define how the AI extracts information from Documents.

        # **NB:** The result of extraction must be a copy of the input Document.

        # Example:
        # result_document = deepcopy(document)

        # The tokenizer will create Annotations objects within the document
        # tokenizer.tokenize(result_document)

        # These Annotations will be the extraction results.
        # At the moment, these Annotations have no Label, which would exclude them from the extraction results.
        # We need to associate the proper Labels to each Annotation, assuming that these exist in our Project.
        # name_label = self.project.get_label_by_name("Name")  # the self.project attribute is derived from Trainer
        # surname_label = self.project.get_label_by_name("Surname")
        # for annotation in result_document.annotations():
        #     for span in annotation.spans:
        #     # Each Annotation contains information about which tokenizer found it.
        #     # In this example, we associate the Label straighforwardly.
        #     # If your regex can produce false positives, you will want to apply some filtering logic here.
        #         if name_tokenizer in span.regex_matching:
        #             annotation.label = name_label
        #             break
        #         elif surname_tokenizer in span.regex_matching:
        #             annotation.label = surname_label
        #             break

        # Suppose we want to extract "A Software Company Ltd.", which does not have a clear regex pattern, but
        # we know it's always the third line in the Document. We can explicitly create an Annotation based on a
        # substring of the Document's text.
        # company_label = self.project.get_label_by_name("Company")
        # company_substring = result_document.split('\n')[2]  # third line of the Document
        # start_offset = result_document.find(company_substring)
        # end_offset = start_offset + len(company_substring)
        # _ = Annotation(document=result_document, label=company_label, spans=[Span(start_offset, end_offset)])

        # The resulting Document has 3 extractions. You can double-check that they are there with:
        # >>> result_document.annotations(use_correct=False)
        # [
        #     Annotation Name (6, 10),
        #     Annotation Surname (20, 23),
        #     Annotation Company (24, 27)
        # ]
        # return result_document

        def check_is_ready(self) -> bool:
            pass

        # define if all components needed for training/prediction are set, for instance, is self.tokenizer set or does
        # self.category contain training and testing Documents.

    # end custom

    import os
    from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID

    YOUR_PROJECT_ID = TEST_PROJECT_ID
    YOUR_CATEGORY_ID = 63
    YOUR_DOCUMENT_ID = TEST_DOCUMENT_ID
    from konfuzio_sdk.trainer.information_extraction import RFExtractionAI
    from konfuzio_sdk.tokenizer.base import ListTokenizer

    # start init_ai
    from konfuzio_sdk.data import Project, Document

    # Initialize Project and the AI
    project = Project(id_=YOUR_PROJECT_ID)  # see https://dev.konfuzio.com/sdk/get_started.html#example-usage
    category = project.get_category_by_id(YOUR_CATEGORY_ID)
    extraction_pipeline = CustomExtractionAI(category)
    # end init_ai
    assert isinstance(extraction_pipeline, CustomExtractionAI)
    project = Project(id_=YOUR_PROJECT_ID, strict_data_validation=False)
    category = project.get_category_by_id(63)
    extraction_pipeline = RFExtractionAI()
    extraction_pipeline.tokenizer = ListTokenizer(tokenizers=[])
    extraction_pipeline.category = category
    # start category
    # provide the categories, training and test data
    extraction_pipeline.documents = extraction_pipeline.category.documents()
    extraction_pipeline.test_documents = extraction_pipeline.category.test_documents()
    # end category
    extraction_pipeline.documents = extraction_pipeline.documents[5:10]
    extraction_pipeline.df_train, extraction_pipeline.label_feature_list = extraction_pipeline.feature_function(
        documents=extraction_pipeline.documents, require_revised_annotations=False
    )
    # start train
    # Train the AI
    extraction_pipeline.fit()

    # Extract a Document
    document = project.get_document_by_id(YOUR_DOCUMENT_ID)
    extraction_result: Document = extraction_pipeline.extract(document=document)

    # Save and load a pickle file for the model
    pickle_model_path = extraction_pipeline.save(output_dir=project.model_folder, include_konfuzio=True)
    extraction_pipeline_loaded = RFExtractionAI.load_model(pickle_model_path)
    # end train
    assert isinstance(extraction_result, Document)
    assert isinstance(extraction_pipeline_loaded, RFExtractionAI)

    os.remove(pickle_model_path)
