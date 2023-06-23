## Paragraph and Sentence Tokenizer

The `ParagraphTokenizer` and `SentenceTokenizer` are used to split a document into paragraphs and sentences 
respectively. They both come with two different modes: `detectron` and `line_distance`. The `detectron` mode uses a 
fine-tuned [Detectron2](https://github.com/facebookresearch/detectron2) model to detect paragraph Annotations. The 
`line_distance` mode uses a rule based approach, and is therefore faster, but tends to be less accurate. In particular, 
it fails to handle documents with two columns. The `detectron` mode is the default. It can also be used together with 
the `create_detectron_labels` setting to create Annotations with the label given by our [Detectron2](https://github.com/facebookresearch/detectron2) model: 
`figure`, `table`, `list`, `text` and `title`.

.. _paragraph-tokenizer-tutorial:

### Paragraph Tokenizer

For example, to tokenize a Document into paragraphs using the `ParagraphTokenizer` in `detectron` mode and the 
`create_detectron_labels` option to use the labels provided by our Detectron model, you can use the following code:

.. literalinclude:: /sdk/boilerplates/test_paragraph_tokenizer.py
   :language: python
   :start-after: start init
   :end-before: end init
   :dedent: 4

The resulting Annotations will look like this:

.. image:: /_static/img/paragraph_tokenizer.png

### Sentence Tokenizer

If you are interested in a more fine-grained tokenization, you can use the `SentenceTokenizer`. It can be used to create 
Annotations for each individual sentence in a text Document. To use it, you can use the following code:

.. literalinclude:: /sdk/boilerplates/test_sentence_tokenizer.py
   :language: python
   :start-after: start import
   :end-before: end import
   :dedent: 4

The resulting Annotations will look like this:

.. image:: /_static/img/sentence_tokenizer.png
