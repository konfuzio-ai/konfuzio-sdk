import logging

import pandas as pd

from konfuzio.load_data import load_pickle
from konfuzio.models_labels_multiclass import DocumentAnnotationMultiClassModel
from konfuzio.tokenizers.whitspace_tokenizers import WhitespaceTokenizer
from konfuzio.wrapper import get_latest_document_model
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.base import ListTokenizer
from konfuzio_sdk.tokenizer.regex import RegexTokenizer

logger = logging.getLogger(__name__)


# TODO: will be moved to SDK
class CertificateTokenizer(RegexTokenizer):
    """Tokenizes text by splitting on whitespace."""

    def __init__(self, *args, **kwargs):
        """Compile the whitespace regex."""
        super().__init__(regex=r'ISO 9001:\d\d\d\d|jfdkjflkdfdf', *args, **kwargs)


class Certificate(DocumentAnnotationMultiClassModel):

    def build(self):
        """Build an DocumentAnnotationMultiClassModel."""
        # Tokenizer selection

        self.tokenizer = ListTokenizer(tokenizers=[WhitespaceTokenizer(), CertificateTokenizer()])
        logger.info('Using WhitespaceTokenizer by default.')

        self.tokenize(documents=self.documents + self.test_documents)
        self.create_candidates_dataset()
        self.train_valid_split()
        self.fit()
        self.fit_label_set_clf()
        if not self.df_test.empty:
            self.evaluate()
        else:
            logger.error('The test set is empty. Skip evaluation for test data.')

        return self

if __name__ == "__main__":
    training_prj = Project(id_=85, update=True)
    logger.info("Project initialized.")
    category = training_prj.get_category_by_id(323)

    doc_model = Certificate(category=category)
    doc_model.build()
    doc_model_path = doc_model.save(output_dir=training_prj.model_folder, include_konfuzio=False)

    # doc_model_path = get_latest_document_model('*.pkl', folder_path=training_prj.model_folder)
    model = load_pickle(doc_model_path)


    logger.info("Model trained.")
    data = []
    for document in category.documents()[:1]:
        evaluation = model.evaluate_extraction_model(document)
        data.append(evaluation)

    df_data = pd.concat(data)
    df_data.to_csv(f'{doc_model_path}.csv')

    # upload_ai_model(model_path, category_ids=[category.id_])
