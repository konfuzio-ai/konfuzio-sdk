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
    from konfuzio_sdk.trainer.document_categorization import AbstractCategorizationAI
    from konfuzio_sdk.data import Page, Category
    from typing import List

    class CustomCategorizationAI(AbstractCategorizationAI):
        def __init__(self, categories: List[Category], *args, **kwargs):
            super().__init__(categories)
            pass

        # initialize key variables required by the custom AI:
        # for instance, self.documents and self.test_documents to train and test the AI on, self.categories to determine
        # which Categories will the AI be able to predict

        def fit(self):
            pass

        # Define architecture and training that the model undergoes, i.e. a NN architecture or a custom hardcoded logic
        # for instance:
        #
        # self.classifier_iterator = build_document_classifier_iterator(
        #             self.documents,
        #             self.train_transforms,
        #             use_image = True,
        #             use_text = False,
        #             device='cpu',
        #         )
        # self.classifier._fit_classifier(self.classifier_iterator, **kwargs)
        #
        # This method is allowed to be implemented as a no-op if you provide the trained model in other ways

        def _categorize_page(self, page: Page) -> Page:
            pass

        # define how the model assigns a Category to a Page.
        # for instance:
        #
        # predicted_category_id, predicted_confidence = self._predict(page_image)
        #
        # for category in self.categories:
        #     if category.id_ == predicted_category_id:
        #         _ = CategoryAnnotation(category=category, confidence=predicted_confidence, page=page)
        #
        # **NB:** The result of extraction must be the input Page with added Categorization attribute `Page.category`

        def save(self, path: str):
            pass

        # define how to save a model in a .pt format – for example, in a way it's defined in the CategorizationAI
        #
        #  data_to_save = {
        #             'tokenizer': self.tokenizer,
        #             'image_preprocessing': self.image_preprocessing,
        #             'image_augmentation': self.image_augmentation,
        #             'text_vocab': self.text_vocab,
        #             'category_vocab': self.category_vocab,
        #             'classifier': self.classifier,
        #             'eval_transforms': self.eval_transforms,
        #             'train_transforms': self.train_transforms,
        #             'model_type': 'CategorizationAI',
        #         }
        # torch.save(data_to_save, path)

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
