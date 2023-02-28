"""Camelot-py based table extractions."""
import os
import logging
from copy import deepcopy

import camelot

from konfuzio_sdk.utils import get_timestamp
from konfuzio_sdk.data import Project, Category, Document, Annotation, Span, AnnotationSet
from konfuzio_sdk.evaluate import Evaluation
from konfuzio_sdk.trainer.information_extraction import Trainer, load_model

logger = logging.getLogger(__name__)


class TableExtractionAI(Trainer):
    """Table Extraction AI using camelot-py."""

    def __init__(
        self,
        category: Category = None,
        *args,
        **kwargs,
    ):
        """Table Extraction AI."""
        logger.info("Initializing Table Extraction AI.")
        super().__init__(category, *args, **kwargs)

        logger.info("Table Extraction AI settings:")
        logger.info(f"{category=}")
        self.category = category

        self.no_label_set_name = None
        self.no_label_name = None

        self.output_dir = None

        self.camelot_func = camelot.read_pdf

    def check_is_ready(self):
        """Check if the classifiers set and trained."""
        if not self.category:
            raise AttributeError(f'{self} requires a Category.')

    @property
    def project(self):
        """Get RFExtractionAI Project."""
        if not self.category:
            raise AttributeError(f'{self} has no Category.')
        return self.category.project

    @property
    def temp_pkl_file_path(self) -> str:
        """Generate a path for temporary pickle file."""
        temp_pkl_file_path = os.path.join(
            self.output_dir, f'{get_timestamp()}_{self.category.name.lower()}_tmp.cloudpickle'
        )
        return temp_pkl_file_path

    @property
    def pkl_file_path(self) -> str:
        """Generate a path for a resulting pickle file."""
        pkl_file_path = os.path.join(self.output_dir, f'{get_timestamp()}_{self.category.name.lower()}.pkl')
        return pkl_file_path

    def evaluate_full(
        self, strict: bool = True, use_training_docs: bool = False, use_view_annotations: bool = True
    ) -> Evaluation:
        """
        Evaluate the full pipeline on the pipeline's Test Documents.

        :param strict: List of documents to extract features from.
        :param use_training_docs: Bool for whether to evaluate on the training documents instead of testing documents.
        :return: Evaluation object.
        """
        eval_list = []
        if not use_training_docs:
            eval_docs = self.test_documents
        else:
            eval_docs = self.documents

        for document in eval_docs:
            predicted_doc = self.extract(document=document)
            eval_list.append((document, predicted_doc))

        full_evaluation = Evaluation(eval_list, strict=strict, use_view_annotations=use_view_annotations)

        return full_evaluation

    def extract(self, document: Document) -> Document:
        """Extract tables from a given Document."""
        logger.info(f"Starting table extraction of {document}.")

        self.check_is_ready()

        inference_document = document  # todo deepcopy
        #inference_document.set_category(self.category)
        #inference_document.get_file(ocr_version=False)

        virtual_annotation_set_id = 1  # counter for across mult. Annotation Set groups of a Label Set

        # define Annotation Set for the Category Label Set
        # default Annotation Set will be always added even if there are no predictions for it
        category_label_set = self.project.get_label_set_by_id(self.category.id_)
        virtual_default_annotation_set = AnnotationSet(
            document=inference_document, label_set=category_label_set, id_=virtual_annotation_set_id
        )

        for page in inference_document.pages():
            label_set = self.project.get_label_set_by_name("A.4.1")  # todo generalize to any table layout
            tables = self.camelot_func(inference_document.file_path, flavor='stream', pages=str(page.number))
            df = tables[0].df.copy()
            df = df.rename(columns=df.iloc[0])
            df = df.drop(df.index[0]).reset_index(drop=True)

            prev_col_name = ""
            for c in df.columns:
                if 'TabSach' in c:
                    new_prev_col_name = c.split(' in ')[0].strip()
                    new_this_col_name = c.split(' TabSach')[0].strip()
                    df = df.rename(columns={prev_col_name: new_prev_col_name, c: new_this_col_name})
                prev_col_name = c

            prev_offset = 0
            for i, row in df.iterrows():
                virtual_annotation_set_id += 1
                virtual_annotation_set = AnnotationSet(
                    document=inference_document, label_set=label_set, id_=virtual_annotation_set_id
                )
                for label_name in df.columns:
                    try:
                        label = self.project.get_label_by_name(label_name)
                    except IndexError:  # todo support all tables, currently skip unsupported table types
                        continue
                    cell_text = row[label_name]
                    if not cell_text:
                        continue
                    try:
                        page.text[prev_offset:].index(cell_text)
                    except ValueError:
                        # for multi-line rows that correspond to merged cells, camelot prints them in reverse order
                        # so we need to go back to the previous line to recover them
                        prev_offset -= 100  # todo replace with character count in a line
                    start_offset = page.start_offset + page.text[prev_offset:].index(cell_text) + prev_offset
                    end_offset = start_offset + len(cell_text)
                    prev_offset = end_offset - page.start_offset
                    span = Span(start_offset=start_offset, end_offset=end_offset)
                    #try:
                    _ = Annotation(
                        document=inference_document,
                        annotation_set=virtual_annotation_set,
                        label_set=label_set,
                        label=label,
                        spans=[span],
                        confidence=1.0,
                        accuracy=table.accuracy,
                    )
                    #except ValueError:  # todo this should not be necessary
                    #    continue

        return inference_document
