import logging
import os
from functools import partial

import pandas as pd

# from konfuzio.load_data import load_pickle
# from konfuzio.models_labels_multiclass import DocumentAnnotationMultiClassModel
from pathos.multiprocessing import ProcessPool

from konfuzio_sdk.data import Project
from konfuzio_sdk.pipelines.extraction_ai import DocumentAnnotationMultiClassModel
from konfuzio_sdk.pipelines.load_data import load_pickle

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    training_prj = Project(id_=1364, update=True)
    logger.info("Project initialized.")
    category = training_prj.get_category_by_id(4127)

    _a = []
    for document in category.documents():
        if document.check_bbox():
            print('good', document)
        else:
            _a.append(document.id_)
            print('bad', document)
    print(_a)
    doc_model = DocumentAnnotationMultiClassModel(
        category=category,
    )

    doc_model.build()
    doc_model_path = doc_model.save(output_dir=training_prj.model_folder, include_konfuzio=True)
    model = load_pickle(doc_model_path)

    def _evaluate(document, model):
        return model.evaluate_extraction_model(document)

    pool = ProcessPool()
    data = pool.map(partial(_evaluate, model=model), category.documents())
    df_data = pd.concat(data)
    df_data.to_csv('fdfdgjl.csv')

    # for document in category.documents():
    #     evaluation = model.evaluate_extraction_model(document)
    #     data.append(evaluation)
#
#
#     # TODO save() unloads documents from project.
#     training_prj = Project(id_=1252)
#     category = training_prj.get_category_by_id(3832)
#
#     logger.info("Model trained.")
#     data = []
#     for document in category.documents():
#         evaluation = model.evaluate_extraction_model(document)
#         data.append(evaluation)
#

#
#     # upload_ai_model(model_path, category_ids=[category.id_])
# 'data_1252/models/2022-04-22-22-35-16_thinksurance api project.pkl'