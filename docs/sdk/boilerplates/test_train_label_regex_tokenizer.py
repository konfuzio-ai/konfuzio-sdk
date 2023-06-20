"""Test code examples for training a Label regex Tokenizer."""
import pytest


def test_train_label_regex_tokenizer():
    """Test train label regex tokenizer."""
    from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID

    YOUR_PROJECT_ID, YOUR_CATEGORY_ID, YOUR_DOCUMENT_ID = TEST_PROJECT_ID, 63, TEST_DOCUMENT_ID

    # start import
    from konfuzio_sdk.data import Project
    from konfuzio_sdk.tokenizer.regex import RegexTokenizer
    from konfuzio_sdk.tokenizer.base import ListTokenizer

    # end import

    # start initialize
    my_project = Project(id_=YOUR_PROJECT_ID)
    category = my_project.get_category_by_id(id_=YOUR_CATEGORY_ID)
    # end initialize

    # start listtokenizer
    tokenizer = ListTokenizer(tokenizers=[])
    # end listtokenizer

    # start label
    label = my_project.get_label_by_name("Lohnart")
    # end label

    # start train
    for regex in label.find_regex(category=category):
        regex_tokenizer = RegexTokenizer(regex=regex)
        tokenizer.tokenizers.append(regex_tokenizer)
    # end train

    # start use
    # You can then use it to create an Annotation for every matching string in a Document.
    document = my_project.get_document_by_id(YOUR_DOCUMENT_ID)
    tokenizer.tokenize(document)
    # end use
    assert len(document.spans()) == 179


@pytest.mark.skip(reason="Copy of test above with uninterrupted code lines.")
def test_train_label_regex_tokenizer_uninterrupted():
    """Repeat of train label regex Tokenizer with uninterrupted code lines."""
    from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID

    YOUR_PROJECT_ID, YOUR_CATEGORY_ID, YOUR_DOCUMENT_ID = TEST_PROJECT_ID, 63, TEST_DOCUMENT_ID

    # start full training
    from konfuzio_sdk.data import Project
    from konfuzio_sdk.tokenizer.regex import RegexTokenizer
    from konfuzio_sdk.tokenizer.base import ListTokenizer

    my_project = Project(id_=YOUR_PROJECT_ID)
    category = my_project.get_category_by_id(id_=YOUR_CATEGORY_ID)

    tokenizer = ListTokenizer(tokenizers=[])

    label = my_project.get_label_by_name("Lohnart")

    for regex in label.find_regex(category=category):
        regex_tokenizer = RegexTokenizer(regex=regex)
        tokenizer.tokenizers.append(regex_tokenizer)

    # You can then use it to create an Annotation for every matching string in a Document.
    document = my_project.get_document_by_id(YOUR_DOCUMENT_ID)
    tokenizer.tokenize(document)
    # end full training
    assert len(document.spans()) == 179
