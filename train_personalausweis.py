import logging
import os

import pandas

from konfuzio_sdk.api import upload_ai_model
from konfuzio_sdk.data import Project
from konfuzio_sdk.evaluate import Evaluation
from konfuzio_sdk.pipelines.extraction_ai import DocumentAnnotationMultiClassModel
from konfuzio_sdk.pipelines.load_data import load_pickle

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

training_prj = Project(id_=2, update=True)
csv_path = os.path.join(training_prj.model_folder, "2022-06-10-13-13-29.csv")
df = pandas.read_csv(csv_path)
e = Evaluation(df)
print(e)
e.label_evaluations()
e.label_evaluations(dataset_status=[2])
e.label_evaluations(dataset_status=[3])