import logging
from typing import List

from konfuzio_sdk.api import upload_ai_model
from konfuzio_sdk.data import Project
from konfuzio_sdk.pipelines.base import ExtractionModel
from konfuzio_sdk.pipelines.extraction_ai import DocumentAnnotationMultiClassModel
from konfuzio_sdk.pipelines.load_data import load_pickle, get_latest_document_model
from konfuzio_sdk.tokenizer.base import ListTokenizer
from konfuzio_sdk.tokenizer.regex import RegexTokenizer

logger = logging.getLogger(__name__)


class ListExtractionAI(ExtractionModel):

    def __init__(self, extraction_ais, category):
        super().__init__()
        self.category=category
        self.extraction_ais = extraction_ais

        # Empty Label and Label Set to create the potential Annotations
        self.no_label = self.category.project.no_label
        self.no_label_set = self.category.project.no_label_set

    def extract(self, *args, **kwargs):
        res_dict = {}
        for extraction_ai in self.extraction_ais:
            res = extraction_ai.extract(*args, **kwargs)
            if set(res.keys()).intersection(set(res_dict.keys())):
                raise ValueError('Label predicted which was already predicted.')
            res_dict.update(res)
        return res_dict


class LabelModel(DocumentAnnotationMultiClassModel):

    def __init__(self, label, category):
        super().__init__(category)
        self.label = label

    def get_y_train(self, annotations) -> List[str]:
        y_train = []
        for annotation in annotations:
            for span in annotation.spans:
                if span.annotation.label == self.label:
                    y_train.append(span.annotation.label.name)  # Label name should no always be set
                else:
                    y_train.append(self.no_label.name)
        return y_train

    def build(self):
        """Build an DocumentAnnotationMultiClassModel."""
        logger.info('Start data_checks()...')
        self.data_checks()
        # logger.info('Start fit_tokenizer()...')
        # self.fit_tokenizer()
        logger.info('Start tokenize()...')
        self.tokenize(documents=self.documents + self.test_documents, multiprocess=False)
        logger.info('Start create_candidates_dataset()...')
        self.create_candidates_dataset()
        logger.info('Start train_valid_split()...')
        self.train_valid_split()
        logger.info('Start fit()...')
        self.fit()
        # logger.info('Start fit_label_set_clf()...')
        # self.fit_label_set_clf()
        logger.info('Start evaluate()...')
        self.evaluate(multiprocess=False)

        return self


if __name__ == "__main__":
    training_prj = Project(id_=24, update=True)
    logger.info("Project initialized.")
    category = training_prj.get_category_by_id(2455)

    labels = [training_prj.get_label_by_name('Rechnungsnummer')]
    # labels = [training_prj.get_label_by_name('Part 2')]
    # labels = [training_prj.get_label_by_name('Issue_Date')]

    extraction_ais = []
    for label in labels:
        # Add Regex tokenizers.
        new_tokenizers = []
        regexes = label.find_regex(category=category, annotations=label.annotations(categories=[category]))
        for regex in regexes:
            new_tokenizers.append(RegexTokenizer(regex))

        # category.project._documents = category.documents()[:1]
        doc_model = LabelModel(category=category, label=label)
        # doc_model.configure(dict(n_nearest_left=10, n_nearest_right=10))
        doc_model.tokenizer = ListTokenizer(tokenizers=new_tokenizers)
        doc_model.build()
        extraction_ais.append(doc_model)
        doc_model.save(output_dir=category.project.model_folder, name=label.name)

    # document = category.documents()[0]
    # extraction_ai = ListExtractionAI(extraction_ais=extraction_ais, category=category)
    # extraction_result = extraction_ai.extract(text=document.text, bbox=document.get_bbox(), pages=document.pages)
    # print(extraction_result)
    #
    # modelpath = extraction_ai.save(include_konfuzio=True)
    # doc_model = load_pickle(modelpath)
    # extraction_result = extraction_ai.extract(text=document.text, bbox=document.get_bbox(), pages=document.pages)
    # print(extraction_result)
