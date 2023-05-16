## Paragraph Tokenizer

The `ParagraphTokenizer` class is a specialized [Tokenizer](https://dev.konfuzio.com/sdk/sourcecode.html#tokenizers) 
designed to create [Annotations](https://dev.konfuzio.com/sdk/sourcecode.html#annotation) splitting a Document into 
meaningful sections. It provides two modes of operation: `detectron` and `line_distance`.

### Parameters

1. `mode`: This parameter determines the mode of operation for the 
[Tokenizer](https://dev.konfuzio.com/sdk/sourcecode.html#tokenizers). It can be `detectron` or `line_distance`. 
In `detectron` mode, the Tokenizer uses a fine-tuned [Detectron2](https://github.com/facebookresearch/detectron2) model 
to assist in Document segmentation. This mode tends to be more accurate but slower as it requires to make an API call to 
the model hosted on Konfuzio servers. The `line_distance` mode, on the other hand, uses a rule-based approach that is 
faster but less accurate, especially with documents having two columns or other complex layouts. The default is 
`detectron`.

2. `line_height_ratio`: Specifies the ratio of the median line height used as a threshold to create new sections. The 
default value is 0.8.

3. `height`: This optional parameter allows you to define a specific line height threshold for creating new sections. 
If set to None, the Tokenizer uses the intelligently calculated height threshold.

4. `create_detectron_labels`: This boolean flag determines whether to apply labels given by the Detectron2 model when 
the Tokenizer is used in `detectron` mode. If the Labels don't exist, they will be created. The potential Labels include 
`figure`, `table`, `list`, `text` and `title`. If this option is set to False, the Tokenizer will create `NO_LABEL` 
Annotations, just like with out other [Tokenizers](https://dev.konfuzio.com/sdk/sourcecode.html#tokenizers). The default 
value is False.

### Example Usage

To tokenize a Document into paragraphs using the `ParagraphTokenizer` in `detectron` mode and the 
`create_detectron_labels` option to use the Labels provided by our Detectron model, you can use the following code:

.. literalinclude:: /sdk/boilerplates/test_paragraph_tokenizer.py
   :language: python
   :lines: 6-7,11-22
   :dedent: 4

The resulting Annotations will look like this:

.. image:: /_static/img/paragraph_tokenizer.png

For a quicker result with a relatively simpler, single-column Document, you can use the `ParagraphTokenizer` in 
`line_distance` mode like this:

.. literalinclude:: /sdk/boilerplates/test_paragraph_tokenizer.py
   :language: python
   :lines: 27-28,32-43
   :dedent: 4

.. image:: /_static/img/line_distance_paragraph_tokenizer.png

## Sentence Tokenizer

The `SentenceTokenizer` class, similar to [`ParagraphTokenizer`](https://dev.konfuzio.com/sdk/tutorials.html#paragraph-tokenizer), 
is a specialized Tokenizer designed to split a Document into sentences. It also provides two modes of operation: 
`detectron` and `line_distance`. And just like the `ParagraphTokenizer`, you can customize the behavior of the Tokenizer 
by passing using the `mode`, `line_height_ratio`, `height` and `create_detectron_labels` parameters. The only difference 
is that after dividing the Document into sections, the Tokenizer will further split each section into sentences.

### Example Usage

To use it, you can use the following code:

.. literalinclude:: /sdk/boilerplates/test_sentence_tokenizer.py
   :language: python
   :lines: 6-7,11-23
   :dedent: 4

The resulting Annotations will look like this:

.. image:: /_static/img/sentence_tokenizer.png
