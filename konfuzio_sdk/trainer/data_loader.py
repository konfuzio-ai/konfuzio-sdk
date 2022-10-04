"""Utility functions for loading data for document categorization models."""
import functools
import logging
import random
from typing import List, Tuple, Dict

from PIL import Image
import torchvision
import torch

from konfuzio_sdk.trainer.tokenization import Vocab, Tokenizer
from konfuzio_sdk.data import Document
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


def get_document_classifier_data(
    documents: List[Document],
    tokenizer: Tokenizer,
    text_vocab: Vocab,
    category_vocab: Vocab,
    use_image: bool,
    use_text: bool,
    shuffle: bool,
    split_ratio: float,
    max_len: int,
):
    """
    Prepare the data necessary for the document classifier.

    For each document we split into pages and from each page we take:
      - the path to an image of the page
      - the tokenized and numericalized text on the page
      - the label (category) of the page
      - the id of the document
      - the page number
    """
    assert use_image or use_text, 'One of either `use_image` or `use_text` needs to be `True`!'

    data = []

    for document in documents:
        project_id = str(document.project.id_)
        doc_info = get_document_classifier_examples(
            document, project_id, tokenizer, text_vocab, category_vocab, max_len, use_image, use_text
        )
        data.extend(zip(*doc_info))

    # shuffles the data
    if shuffle:
        random.shuffle(data)

    # creates a split if necessary
    n_split_examples = int(len(data) * split_ratio)
    # if we wanted at least some examples in the split, ensure we always have at least 1
    if split_ratio > 0.0:
        n_split_examples = max(1, n_split_examples)
    split_data = data[:n_split_examples]
    _data = data[n_split_examples:]

    return _data, split_data


def build_document_classifier_iterator(
    data: List,
    transforms: torchvision.transforms,
    text_vocab: Vocab,
    use_image: bool,
    use_text: bool,
    shuffle: bool,
    batch_size: int,
    device: torch.device = 'cpu',
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build the iterators for the data list."""
    logger.debug("build_document_classifier_iterator")

    def collate(batch, transforms) -> Dict[str, torch.LongTensor]:
        image_path, text, label, doc_id, page_num = zip(*batch)
        if use_image:
            # if we are using images, open as PIL images, apply transforms and place on GPU
            image = [Image.open(path) for path in image_path]
            image = torch.stack([transforms(img) for img in image], dim=0).to(device)
            image = image.to(device)
        else:
            # if not using images then just set to None
            image = None
        if use_text:
            # if we are using text, batch and pad the already tokenized and numericalized text and place on GPU
            text = torch.nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value=text_vocab.pad_idx)
            text = text.to(device)
        else:
            text = None
        # also place label on GPU
        # doc_id and page_num do not need to be placed on GPU
        label = torch.cat(label).to(device)
        doc_id = torch.cat(doc_id)
        page_num = torch.cat(page_num)
        # pack everything up in a batch dictionary
        batch = {'image': image, 'text': text, 'label': label, 'doc_id': doc_id, 'page_num': page_num}
        return batch

    # get the collate functions with the appropriate transforms
    data_collate = functools.partial(collate, transforms=transforms)

    # build the iterators
    iterator = DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collate)

    return iterator


def build_document_classifier_iterators(
    train_documents: List[Document],
    test_documents: List[Document],
    tokenizer: Tokenizer,
    eval_transforms: torchvision.transforms,
    train_transforms: torchvision.transforms,
    text_vocab: Vocab,
    category_vocab: Vocab,
    use_image: bool,
    use_text: bool,
    valid_ratio: float = 0.2,
    batch_size: int = 16,
    max_len: int = 50,
    device: torch.device = 'cpu',
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build the iterators for the document classifier."""
    assert use_image or use_text, 'One of either `use_image` or `use_text` needs to be `True`!'

    logger.info('building document classifier iterators')

    # get data (list of examples) from Documents
    train_data, valid_data = get_document_classifier_data(
        train_documents,
        tokenizer,
        text_vocab,
        category_vocab,
        use_image,
        use_text,
        shuffle=True,
        split_ratio=valid_ratio,
        max_len=max_len,
    )

    test_data, _ = get_document_classifier_data(
        test_documents,
        tokenizer,
        text_vocab,
        category_vocab,
        use_image,
        use_text,
        shuffle=False,
        split_ratio=0,
        max_len=max_len,
    )

    logger.info(f'{len(train_data)} training examples')
    logger.info(f'{len(valid_data)} validation examples')
    logger.info(f'{len(test_data)} testing examples')

    train_iterator = build_document_classifier_iterator(
        train_data,
        train_transforms,
        text_vocab,
        use_image,
        use_text,
        shuffle=True,
        batch_size=batch_size,
        device=device,
    )
    valid_iterator = build_document_classifier_iterator(
        valid_data,
        eval_transforms,
        text_vocab,
        use_image,
        use_text,
        shuffle=False,
        batch_size=batch_size,
        device=device,
    )
    test_iterator = build_document_classifier_iterator(
        test_data, eval_transforms, text_vocab, use_image, use_text, shuffle=False, batch_size=batch_size, device=device
    )

    return train_iterator, valid_iterator, test_iterator


def get_document_classifier_examples(
    document: Document,
    project_id: str,
    tokenizer: Tokenizer,
    text_vocab: Vocab,
    category_vocab: Vocab,
    max_len: int,
    use_image: bool,
    use_text: bool,
):
    """Get the per document examples for the document classifier."""
    document_image_paths = []
    document_tokens = []
    document_labels = []
    document_ids = []
    document_page_numbers = []

    # validate the data for the document
    if use_image:
        document.get_images()  # gets the images if they do not exist
        image_paths = [page.image_path for page in document.pages()]  # gets the paths to the images
        assert len(image_paths) > 0, f'No images found for document {document.id_}'
        if not use_text:  # if only using image OR text then make the one not used a list of None
            page_texts = [None] * len(image_paths)
    if use_text:
        page_texts = document.text.split('\f')
        assert len(page_texts) > 0, f'No text found for document {document.id_}'
        if not use_image:  # if only using image OR text then make the one not used a list of None
            image_paths = [None] * len(page_texts)

    # check we have the same number of images and text pages
    # only useful when we have both an image and a text module
    assert len(image_paths) == len(
        page_texts
    ), f'No. of images ({len(image_paths)}) != No. of pages {len(page_texts)} for document {document.id_}'

    for page_number, (image_path, page_text) in enumerate(zip(image_paths, page_texts)):
        if use_image:
            # if using an image module, store the path to the image
            document_image_paths.append(image_path)
        else:
            # if not using image module then don't need the image paths
            # so we just have a list of None to keep the lists the same length
            document_image_paths.append(None)
        if use_text:
            # if using a text module, tokenize the page, trim to max length and then numericalize
            page_tokens = tokenizer.get_tokens(page_text)[:max_len]
            document_tokens.append(torch.LongTensor([text_vocab.stoi(t) for t in page_tokens]))
        else:
            # if not using text module then don't need the tokens
            # so we just have a list of None to keep the lists the same length
            document_tokens.append(None)
        # append the label (category), the document's id number and the page number of each page
        document_labels.append(torch.LongTensor([category_vocab.stoi(project_id)]))
        document_ids.append(torch.LongTensor([document.id_]))
        document_page_numbers.append(torch.LongTensor([page_number]))

    return document_image_paths, document_tokens, document_labels, document_ids, document_page_numbers


def get_document_template_classifier_data(
    documents: List[Document],
    tokenizer: Tokenizer,
    text_vocab: Vocab,
    category_vocab: Vocab,
    use_image: bool,
    use_text: bool,
    shuffle: bool,
    split_ratio: float,
    max_len: int,
):
    """
    Prepare the data necessary for the document classifier.

    For each document we split into pages and from each page we take:
      - the path to an image of the page
      - the tokenized and numericalized text on the page
      - the label (category) of the page
      - the id of the document
      - the page number
    """
    assert use_image or use_text, 'One of either `use_image` or `use_text` needs to be `True`!'

    data = []

    for document in documents:
        # get document classification (defined by the category template)
        meta_data = document.project.meta_data
        template_id = [m['category_template'] for m in meta_data if m['id'] == document.id_]
        assert len(template_id) == 1
        template_id = str(template_id[0]) if template_id[0] else 'NO_LABEL'

        doc_info = get_document_classifier_examples(
            document, template_id, tokenizer, text_vocab, category_vocab, max_len, use_image, use_text
        )
        data.extend(zip(*doc_info))

    # shuffles the data
    if shuffle:
        random.shuffle(data)

    # creates a split if necessary
    n_split_examples = int(len(data) * split_ratio)
    # if we wanted at least some examples in the split, ensure we always have at least 1
    if split_ratio > 0.0:
        n_split_examples = max(1, n_split_examples)
    split_data = data[:n_split_examples]
    _data = data[n_split_examples:]

    return _data, split_data


def build_document_template_classifier_iterators(
    train_documents: List[Document],
    test_documents: List[Document],
    tokenizer: Tokenizer,
    eval_transforms: torchvision.transforms,
    train_transforms: torchvision.transforms,
    text_vocab: Vocab,
    category_vocab: Vocab,
    use_image: bool,
    use_text: bool,
    valid_ratio: float = 0.2,
    batch_size: int = 16,
    max_len: int = 50,
    device: torch.device = 'cpu',
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build the iterators for the document classifier."""
    assert use_image or use_text, 'One of either `use_image` or `use_text` needs to be `True`!'

    logger.info('building document classifier iterators')

    # get data (list of examples) from Documents
    train_data, valid_data = get_document_template_classifier_data(
        train_documents,
        tokenizer,
        text_vocab,
        category_vocab,
        use_image,
        use_text,
        shuffle=True,
        split_ratio=valid_ratio,
        max_len=max_len,
    )

    test_data, _ = get_document_template_classifier_data(
        test_documents,
        tokenizer,
        text_vocab,
        category_vocab,
        use_image,
        use_text,
        shuffle=False,
        split_ratio=0,
        max_len=max_len,
    )

    logger.info(f'{len(train_data)} training examples')
    logger.info(f'{len(valid_data)} validation examples')
    logger.info(f'{len(test_data)} testing examples')

    train_iterator = build_document_classifier_iterator(
        train_data,
        train_transforms,
        text_vocab,
        use_image,
        use_text,
        shuffle=True,
        batch_size=batch_size,
        device=device,
    )
    valid_iterator = build_document_classifier_iterator(
        valid_data,
        eval_transforms,
        text_vocab,
        use_image,
        use_text,
        shuffle=False,
        batch_size=batch_size,
        device=device,
    )
    test_iterator = build_document_classifier_iterator(
        test_data, eval_transforms, text_vocab, use_image, use_text, shuffle=False, batch_size=batch_size, device=device
    )

    return train_iterator, valid_iterator, test_iterator
