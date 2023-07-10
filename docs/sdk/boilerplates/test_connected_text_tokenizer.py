"""Test a ConnectedTextTokenizer instance."""


def test_connected_text_tokenizer():
    """Test a ConnectedTextTokenizer."""
    from konfuzio_sdk.samples import LocalTextProject
    from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer

    project = LocalTextProject()
    tokenizer = ConnectedTextTokenizer()
    YOUR_DOCUMENT_ID = 9
    test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)
    assert (
        test_document.text == "Hi all,\nI like bread.\n\fI hope to get everything done soon.\n\fMorning,\n\fI'm glad "
        "to see you.\n\fMorning,"
    )
    assert test_document.spans() == []
    # Start tokenize
    # before tokenization
    test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)
    test_document.text

    # output: "Hi all,\nI like bread.\n\fI hope to get everything done soon.\n\fMorning,\n\fI'm glad to see you."
    #             "\n\fMorning,"

    test_document.spans()

    # output: []

    test_document = tokenizer.tokenize(test_document)

    # after tokenization
    test_document.spans()

    # output: [Span (0, 7), Span (8, 21), Span (22, 58), Span (59, 68), Span (69, 90), Span (91, 100)]
    # End tokenize
    assert len(test_document.spans()) == 6

    # Start string
    test_document.spans()[0].offset_string

    # output: "Hi all,"
    # End string
    assert test_document.spans()[0].offset_string == 'Hi all,'
