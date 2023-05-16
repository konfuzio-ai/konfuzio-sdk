## Paragraph and Sentence Tokenizer

The `ParagraphTokenizer` class is a specialized tokenizer designed to split a document into meaningful sections including
paragraphs, titles, lists, figures, and tables. It and provides two modes of operation: `detectron` and `line_distance`.

### Parameters

1. `mode`: This parameter determines the mode of operation for the tokenizer. It can be 'detectron' or 'line_distance'. 
In 'detectron' mode, the Tokenizer uses a fine-tuned [Detectron2](https://github.com/facebookresearch/detectron2) model 
to assist in document segmentation. This mode tends to be more accurate but slower as it requires to make an API call to 
the model hosted on Konfuzio servers. The 'line_distance' mode, on the other hand, uses a rule-based approach that is 
faster but less accurate, especially with documents having two columns or other complex layouts. The default is 
'detectron'.

2. `line_height_ratio`: Specifies the ratio of the median line height used as a threshold to create new sections. The 
default value is 0.8.

3. `height`: This optional parameter allows you to define a specific line height threshold for creating new sections. 
If set to None, the tokenizer uses the intelligently calculated line height ratio.

4. `create_detectron_labels`: This boolean flag determines whether to apply labels given by the Detectron2 model when 
the Tokenizer is used in `detectron` mode. If the Labels don't exist, they will be created. The potential Labels include 
`figure`, `table`, `list`, `text` and `title`. If this option is set to False, the tokenizer will create `NO_LABEL` 
Annotations, just like . The default value is False.

### Example Usage

To tokenize a Document into paragraphs using the `ParagraphTokenizer` in `detectron` mode and the 
`create_detectron_labels` option to use the Labels provided by our Detectron model, you can use the following code:

.. literalinclude:: /sdk/boilerplates/test_paragraph_tokenizer.py
   :language: python
   :lines: 6-7,11-22
   :dedent: 4

The resulting Annotations will look like this:

.. image:: /_static/img/paragraph_tokenizer.png



.. image:: /_static/img/line_distance_paragraph_tokenizer.png


### Sentence Tokenizer

If you are interested in a more fine grained tokenization, you can use the `SentenceTokenizer`. It can be used to create 
Annotations for each individual sentence in a text Document. To use it, you can use the following code:

.. literalinclude:: /sdk/boilerplates/test_sentence_tokenizer.py
   :language: python
   :lines: 6-7,11-23
   :dedent: 4

The resulting Annotations will look like this:

.. image:: /_static/img/sentence_tokenizer.png
