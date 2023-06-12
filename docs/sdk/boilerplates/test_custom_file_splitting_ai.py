"""Test code example for the Custom File Splitting AI creation tutorial."""


def test_custom_file_splitting_ai():
    """Test creating a custom File Splitting AI."""
    # start class
    from konfuzio_sdk.trainer.file_splitting import AbstractFileSplittingModel
    from konfuzio_sdk.data import Page

    class CustomFileSplittingModel(AbstractFileSplittingModel):
        def __init__(self, *args, **kwargs):
            pass

        # initialize key variables required by the custom AI
        # for instance, self.categories to determine which Categories will be used for training the AI, self.documents
        # and self.test_documents to define training and testing Documents, self.tokenizer for a Tokenizer that will
        # be used in processing the Documents

        def fit(self):
            pass

        # Define architecture and training that the model undergoes, i.e. a NN architecture or a custom hardcoded logic
        # This method is allowed to be implemented as a no-op if you provide the trained model in other ways

        def predict(self, page: Page) -> Page:
            pass

        # Define how the model determines a split point for a Page
        # **NB:** The classification needs to be run on the Page level, not the Document level – the result of
        # classification is reflected in `is_first_page` attribute value, which is unique to the Page class and is not
        # present in Document class. Pages with `is_first_page = True` become potential splitting points, thus, each new
        # sub-Document has a Page predicted as first as its starting point.

        def check_is_ready(self) -> bool:
            pass

        # define if all components needed for training/prediction are set

    # end class
    CustomFileSplittingModel
