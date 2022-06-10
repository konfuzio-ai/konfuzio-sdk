import logging

from konfuzio_sdk.api import upload_ai_model
from konfuzio_sdk.data import Project
from konfuzio_sdk.pipelines.extraction_ai import DocumentAnnotationMultiClassModel
from konfuzio_sdk.pipelines.load_data import load_pickle

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    training_prj = Project(id_=1364, update=True)
    logger.info("Project initialized.")
    category = training_prj.get_category_by_id(4127)

    doc_model = DocumentAnnotationMultiClassModel(category=category)
    doc_model.configure({'tokenizer_regex_combination': True, 'n_nearest_left': 5, 'n_nearest_right': 10})
    doc_model.build()
    doc_model_path = doc_model.save(output_dir=training_prj.model_folder, include_konfuzio=True)
    upload_ai_model(doc_model_path, category_ids=[category.id_])
