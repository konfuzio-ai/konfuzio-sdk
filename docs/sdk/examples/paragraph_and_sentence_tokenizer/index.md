## Paragraph Tokenizer

The `ParagraphTokenizer` class is a specialized [Tokenizer](https://dev.konfuzio.com/sdk/sourcecode.html#tokenizers) 
designed to create [Annotations](https://dev.konfuzio.com/sdk/sourcecode.html#annotation) splitting a Document into 
meaningful sections. It provides two modes of operation: `detectron` and `line_distance`.

1. `mode`: This parameter determines the mode of operation for the 
[Tokenizer](https://dev.konfuzio.com/sdk/sourcecode.html#tokenizers). It can be `detectron` or `line_distance`. 
In `detectron` mode, the Tokenizer uses a fine-tuned [Detectron2](https://github.com/facebookresearch/detectron2) model 
to assist in Document segmentation. This mode tends to be more accurate but slower as it requires to make an API call to 
the model hosted on Konfuzio servers. The `line_distance` mode, on the other hand, uses a rule-based approach that is 
faster but less accurate, especially with documents having two columns or other complex layouts. The default is 
`detectron`.

### Line Distance Approach

The line_distance approach offers a straightforward, efficient way to segment documents based on line heights, proving 
especially useful for simple, single-column formats. Despite its limitations with complex layouts, its fast processing 
and relatively accurate results make it a practical choice for tasks where speed is a priority and the document 
structure isn't overly complicated. Essentially, it serves as a quick and easy solution for basic document analysis.

#### Parameters

1. `line_height_ratio`: Specifies the ratio of the median line height used as a threshold to create a new paragraph when 
using the Tokenizer in `line_distance` mode. The default value is 0.8. If you notice that the Tokenizer is not creating 
new paragraphs when it should, you can try lowering this value. Alternatively, if the Tokenizer is creating too many 
paragraphs, you can try increasing this value.

2. `height`: This optional parameter allows you to define a specific line height threshold for creating new paragraphs. 
If set to None, the Tokenizer uses the intelligently calculated height threshold.

For a quicker result with a relatively simpler, single-column Document, you can use the `ParagraphTokenizer` in 
`line_distance` mode like this:

.. literalinclude:: /sdk/boilerplates/test_paragraph_tokenizer.py
   :language: python
   :lines: 29-30,34-36,38-47
   :dedent: 4

.. image:: /_static/img/line_distance_paragraph_tokenizer.png

While the line_distance approach offers efficient document segmentation for simpler formats, it can struggle with complex 
layouts and diverse document elements. The computer vision approach, using tools like Detectron2, adapts to a wide range 
of layouts and provides precise segmentation even in complex scenarios. Although it may require more processing power, 
the significant improvement in accuracy and comprehensiveness makes it a powerful upgrade for high-quality document analysis.

### Computer Vision Approach

With the computer vision approach, we can create labels, identify figures, tables, lists, texts, and titles, thereby giving 
us a comprehensive understanding of the document's structure. The opportunity to apply machine learning in the segmentation 
process allows us to push the boundaries of what's possible in document analysis.

Using the computer vision approach might require more processing power and might be slower compared to the line_distance 
approach, but the significant leap in the quality and comprehensiveness of the output makes it a powerful tool for any 
organization that values precision and quality.

#### Parameters

1. `create_detectron_labels`: This boolean flag determines whether to apply labels given by the Detectron2 model when 
the Tokenizer is used in `detectron` mode. If the Labels don't exist, they will be created. The potential Labels include 
`figure`, `table`, `list`, `text` and `title`. If this option is set to False, the Tokenizer will create `NO_LABEL` 
Annotations, just like with out other [Tokenizers](https://dev.konfuzio.com/sdk/sourcecode.html#tokenizers). The default 
value is False.

To tokenize a Document into paragraphs using the `ParagraphTokenizer` in `detectron` mode and the 
`create_detectron_labels` option to use the Labels provided by our Detectron model, you can use the following code:

.. literalinclude:: /sdk/boilerplates/test_paragraph_tokenizer.py
   :language: python
   :lines: 6-7,11-13,15-24
   :dedent: 4

The resulting Annotations will look like this:

.. image:: /_static/img/paragraph_tokenizer.png

Once you have a grasp on how to implement and utilize the ParagraphTokenizer, you open a new world of possibilities. 
The segmentation of documents doesn't stop at the paragraph level. On the contrary, it is just the beginning. Let's 
embark on a journey to dive deeper into the document's structure and semantics with the SentenceTokenizer.

## Sentence Tokenizer

The `SentenceTokenizer` class, akin to the [`ParagraphTokenizer`](https://dev.konfuzio.com/sdk/tutorials.html#paragraph-tokenizer), 
is a specialized Tokenizer designed to split a Document into sentences. It also provides two modes of operation: 
`detectron` and `line_distance`. And just like the `ParagraphTokenizer`, you can customize the behavior of the Tokenizer 
by passing using the `mode`, `line_height_ratio`, `height` and `create_detectron_labels` parameters. The distinguishing 
feature of the `SentenceTokenizer` is that it will split the Document into sentences, not paragraphs. 

### Example Usage

To use it, you can use the following code:

.. literalinclude:: /sdk/boilerplates/test_sentence_tokenizer.py
   :language: python
   :lines: 6-7,11-13,15-24
   :dedent: 4

The resulting Annotations will look like this:

.. image:: /_static/img/sentence_tokenizer.png
