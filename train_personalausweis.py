import logging
import os
from typing import List

import pandas

from konfuzio_sdk.api import upload_ai_model
from konfuzio_sdk.data import Project
from konfuzio_sdk.evaluate import Evaluation
from konfuzio_sdk.pipelines.extraction_ai import DocumentAnnotationMultiClassModel
from konfuzio_sdk.pipelines.load_data import load_pickle
from konfuzio_sdk.api import upload_ai_model
from konfuzio_sdk.data import Project
from konfuzio_sdk.pipelines.base import ExtractionModel
from konfuzio_sdk.pipelines.extraction_ai import DocumentAnnotationMultiClassModel
from konfuzio_sdk.pipelines.load_data import load_pickle, get_latest_document_model
from konfuzio_sdk.tokenizer.base import ListTokenizer
from konfuzio_sdk.tokenizer.regex import RegexTokenizer
logger = logging.getLogger(__name__)

# if __name__ == "__main__":

#     logger.info("Project initialized.")
#     category = training_prj.get_category_by_id(14)
#     # category.label_sets[0].labels = [training_prj.get_label_by_name('PLZ')]
#     doc_model = DocumentAnnotationMultiClassModel(category=category)
#     doc_model.configure({'tokenizer_regex_combination': True})
#     doc_model.build()
#     doc_model_path = doc_model.save(output_dir=training_prj.model_folder, include_konfuzio=True)
#     upload_ai_model(doc_model_path, category_ids=[category.id_])
#

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


training_prj = Project(id_=2, update=True)
category = training_prj.get_category_by_id(id_=14)

labels = [
    training_prj.get_label_by_name('Gültig Bis'),
]

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


# # Merging code.
#
# extraction_ais = []
# for label in labels:
#     label_model_path = get_latest_document_model(f'*{label.name}.pkl', folder_path=category.project.model_folder)
#     label_model = load_pickle(label_model_path)
#     extraction_ais.append(label_model)
# extraction_ai = ListExtractionAI(extraction_ais=extraction_ais, category=category)
# modelpath = extraction_ai.save(include_konfuzio=True, output_dir=category.project.model_folder)
# upload_ai_model(ai_model_path=modelpath, category_ids=[category.id_])
# print(modelpath)

# csv_path = os.path.join(training_prj.model_folder, "2022-06-10-13-13-29.csv")
# df = pandas.read_csv(csv_path)
# e = Evaluation(df)
# print(e)
# e.label_evaluations()
# e.label_evaluations(dataset_status=[2])
# e.label_evaluations(dataset_status=[3])