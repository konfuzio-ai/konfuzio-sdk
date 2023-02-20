"""Test a code example for the File Splitting section of the documentation."""
import logging

from typing import List

from konfuzio_sdk.data import Page, Category, Project
from konfuzio_sdk.trainer.file_splitting import AbstractFileSplittingModel, SplittingAI
from konfuzio_sdk.trainer.information_extraction import load_model
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer

logger = logging.getLogger(__name__)


class ContextAwareFileSplittingModel(AbstractFileSplittingModel):
    """
    A File Splitting Model that uses a context-aware logic.

    Context-aware logic implies a rule-based approach that looks for common strings between the first Pages of all
    Category's Documents.
    """

    def __init__(self, categories: List[Category], tokenizer, *args, **kwargs):
        """
        Initialize the Context Aware File Splitting Model.

        :param categories: A list of Categories to run training/prediction of the model on.
        :type categories: List[Category]
        :param tokenizer: Tokenizer used for processing Documents on fitting when searching for exclusive first-page
        strings.
        :raises ValueError: When an empty list of Categories is passed into categories argument.
        :raises ValueError: When a list passed into categories contains elements other than Categories.
        :raises ValueError: When a list passed into categories contains at least one Category with no Documents or test
        Documents.
        """
        super().__init__(categories=categories)
        self.output_dir = self.project.model_folder
        self.tokenizer = tokenizer
        self.requires_text = True
        self.requires_images = False

    def fit(self, allow_empty_categories: bool = False, *args, **kwargs):
        """
        Gather the strings exclusive for first Pages in a given stream of Documents.

        Exclusive means that each of these strings appear only on first Pages of Documents within a Category.

        :param allow_empty_categories: To allow returning empty list for a Category if no exclusive first-page strings
        were found during fitting (which means prediction would be impossible for a Category).
        :type allow_empty_categories: bool
        :raises ValueError: When allow_empty_categories is False and no exclusive first-page strings were found for
        at least one Category.
        """
        for category in self.categories:
            # method exclusive_first_page_strings fetches a set of first-page strings exclusive among the Documents
            # of a given Category. they can be found in _exclusive_first_page_strings attribute of a Category after
            # the method has been run. this is needed so that the information remains even if local variable
            # cur_first_page_strings is lost.
            cur_first_page_strings = category.exclusive_first_page_strings(tokenizer=self.tokenizer)
            if not cur_first_page_strings:
                if allow_empty_categories:
                    logger.warning(
                        f'No exclusive first-page strings were found for {category}, so it will not be used '
                        f'at prediction.'
                    )
                else:
                    raise ValueError(f'No exclusive first-page strings were found for {category}.')

    def predict(self, page: Page) -> Page:
        """
        Predict a Page as first or non-first.

        :param page: A Page to receive first or non-first label.
        :type page: Page
        :return: A Page with a newly predicted is_first_page attribute.

        >>> model.predict(
        >>>    model.tokenizer.tokenize(Project(id_=YOUR_PROJECT_ID).get_document_by_id(YOUR_DOCUMENT_ID)
        >>>     ).pages()[0]).is_first_page
        True
        """
        self.check_is_ready()
        page.is_first_page = False
        for category in self.categories:
            cur_first_page_strings = category.exclusive_first_page_strings(tokenizer=self.tokenizer)
            intersection = {span.offset_string for span in page.spans()}.intersection(cur_first_page_strings)
            if len(intersection) > 0:
                page.is_first_page = True
                break
        page.is_first_page_confidence = 1
        return page

    def check_is_ready(self):
        """
        Check File Splitting Model is ready for inference.

        :raises AttributeError: When no Tokenizer or no Categories were passed.
        :raises ValueError: When no Categories have _exclusive_first_page_strings.

        >>> model.check_is_ready()
        True
        """
        if self.tokenizer is None:
            raise AttributeError(f'{self} missing Tokenizer.')

        if not self.categories:
            raise AttributeError(f'{self} requires Categories.')

        empty_first_page_strings = [
            category
            for category in self.categories
            if not category.exclusive_first_page_strings(tokenizer=self.tokenizer)
        ]
        if len(empty_first_page_strings) == len(self.categories):
            raise ValueError(
                f"Cannot run prediction as none of the Categories in {self.project} have "
                f"_exclusive_first_page_strings."
            )


# initialize a Project and fetch a test Document of your choice
YOUR_PROJECT_ID = 46
project = Project(id_=YOUR_PROJECT_ID)
YOUR_DOCUMENT_ID = 44865
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)

# initialize a Context Aware File Splitting Model and fit it

file_splitting_model = ContextAwareFileSplittingModel(categories=project.categories, tokenizer=ConnectedTextTokenizer())

# for an example run, you can take only a slice of training documents to make fitting faster
file_splitting_model.documents = file_splitting_model.documents[:10]

file_splitting_model.fit(allow_empty_categories=True)

# save the model
save_path = file_splitting_model.save(include_konfuzio=True)

# run the prediction
for page in test_document.pages():
    pred = file_splitting_model.predict(page)
    if pred.is_first_page:
        print('Page {} is predicted as the first.'.format(page.number))
    else:
        print('Page {} is predicted as the non-first.'.format(page.number))

# usage with the Splitting AI – you can load a pre-saved model or pass an initialized instance as the input
# in this example, we load a previously saved one
model = load_model(save_path)

# initialize the Splitting AI
splitting_ai = SplittingAI(model)

# Splitting AI is a more high-level interface to Context Aware File Splitting Model and any other models that can be
# developed for file-splitting purposes. It takes a Document as an input, rather than individual Pages, because it
# utilizes page-level prediction of possible split points and returns Document or Documents with changes depending on
# the prediction mode.

# Splitting AI can be run in two modes: returning a list of Sub-Documents as the result of the input Document
# splitting or returning a copy of the input Document with Pages predicted as first having an attribute
# "is_first_page". The flag "return_pages" has to be True for the latter; let's use it
new_document = splitting_ai.propose_split_documents(test_document, return_pages=True)
print(new_document)
# output: [predicted_document]

for page in new_document[0].pages():
    if page.is_first_page:
        print('Page {} is predicted as the first.'.format(page.number))
    else:
        print('Page {} is predicted as the non-first.'.format(page.number))


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        extraglobs={
            'model': model,
        }
    )
