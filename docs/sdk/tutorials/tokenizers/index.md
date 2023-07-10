.. _tokenization-tutorials:
## Tokenization

### WhitespaceTokenizer

The [WhitespaceTokenizer](../../sourcecode.html#konfuzio_sdk.tokenizer.regex.WhitespaceTokenizer), 
part of the Konfuzio SDK, is a foundational tool in natural language processing (NLP). It segments text into smaller 
units called tokens, using the white spaces between words as natural delimiters. This approach is simple, effective, and 
ideal for basic tokenization tasks.

This Tokenizer functions through a regular expression (regex) that identifies sequences of characters unbroken by white 
space. For instance, the text "street Name 1-2b," would be segmented into the tokens "street", "Name", and "1-2b,".

Despite its simplicity, the `WhitespaceTokenizer` is a robust tool for many NLP scenarios. Its uncomplicated design makes 
it an excellent default choice for tasks that do not require advanced tokenization strategies.

#### Example Usage: Getting Word Bounding Box (BBox) for a Document

In this tutorial, we will walk through how to extract the bounding box ([BBox](https://dev.konfuzio.com/sdk/sourcecode.html#bbox)) 
for words in a Document, rather than for individual characters, using the Konfuzio SDK. This process involves the use of 
the `WhitespaceTokenizer` from the Konfuzio SDK to tokenize the Document and identify word-level Spans, which can then 
be visualized or used to extract BBox information.

##### Prerequisites

- You will need to have the Konfuzio SDK installed.
- You should have access to a Project on the Konfuzio platform.

##### Preview of Result

<img src="https://github.com/konfuzio-ai/konfuzio-sdk/assets/2879188/5f7a8501-cd89-487d-a332-0703f3c35fc8" data-canonical-src="https://github.com/konfuzio-ai/konfuzio-sdk/assets/2879188/5f7a8501-cd89-487d-a332-0703f3c35fc8" width="200" height="400" />

##### Steps

.. collapse:: Full code

   .. literalinclude:: /sdk/boilerplates/test_word_bboxes.py
      :language: python
      :start-after: start full word_bboxes
      :end-before: end full word_bboxes
      :dedent: 4

<br/>
1. **Import necessary modules**:

   .. literalinclude:: /sdk/boilerplates/test_word_bboxes.py
      :language: python
      :start-after: start import
      :end-before: end import
      :dedent: 4

2. **Initialize your Project**:

   This involves creating a Project instance with the appropriate ID.

   .. literalinclude:: /sdk/boilerplates/test_word_bboxes.py
      :language: python
      :start-after: start project
      :end-before: end project
      :dedent: 4

3. **Retrieve a Document from your Project**:

   .. literalinclude:: /sdk/boilerplates/test_word_bboxes.py
      :language: python
      :start-after: start document
      :end-before: end document
      :dedent: 4

4. **Create a copy of your Document without Annotations**:

   .. literalinclude:: /sdk/boilerplates/test_word_bboxes.py
      :language: python
      :start-after: start copy
      :end-before: end copy
      :dedent: 4

5. **Tokenize the Document**:

   This process involves splitting the Document into word-level Spans using the `WhitespaceTokenizer`.

   .. literalinclude:: /sdk/boilerplates/test_word_bboxes.py
      :language: python
      :start-after: start tokenize
      :end-before: end tokenize
      :dedent: 4

6. **Visualize all word-level Annotations**:

   After getting the bounding box for all Spans, you might want to visually check the results to make sure the bounding 
   boxes are correctly assigned. Here's how you can do it:

   .. literalinclude:: /sdk/boilerplates/test_word_bboxes.py
      :language: python
      :start-after: start image
      :end-before: end image
      :dedent: 4

   .. image:: /sdk/tutorials/tokenizers/word-bboxes.png

   This will display an image of the Document with all word-level Annotations. The image may look a bit messy with all 
   the Labels.

7. **Get bounding box for all Spans**:

   You can retrieve bounding boxes for all word-level Spans using the following code:

   .. literalinclude:: /sdk/boilerplates/test_word_bboxes.py
      :language: python
      :start-after: start spans
      :end-before: end spans
      :dedent: 4

   Each bounding box (`Bbox`) in the list corresponds to a specific word and is defined by four coordinates: x0 and y0 
   specify the coordinates of the bottom left corner, while x1 and y1 mark the coordinates of the top right corner, 
   thereby specifying the box's position and dimensions on the Document Page.


### Training a Label-Specific Regex Tokenizer

**Pro Tip**: Read our technical blog post [Automated Regex](https://helm-nagel.com/Automated-Regex-Generation-based-on-examples) 
to find out how we use Regex to detect outliers in our annotated data.

The `konfuzio_sdk` package offers many tools for tokenization tasks. For more complex scenarios, like identifying intricate 
Annotation strings, it allows for the training of a custom Regex Tokenizer. This can often be a more effective approach 
than relying on a basic `WhitespaceTokenizer`.

#### Example Usage

In this example, you will see how to find regex expressions that match with occurrences of the "Lohnart" Label in the 
training data. 

.. collapse:: Full code

   .. literalinclude:: /sdk/boilerplates/test_train_label_regex_tokenizer.py
      :language: python
      :start-after: start full training
      :end-before: end full training
      :dedent: 4

<br/>
1. **Import necessary modules**:

   .. literalinclude:: /sdk/boilerplates/test_train_label_regex_tokenizer.py
      :language: python
      :start-after: start import
      :end-before: end import
      :dedent: 4

2. **Initialize your Project and retrieve the Category**:

   This involves creating a Project instance with the appropriate Project ID and retrieving the relevant Category with 
   the Category ID.

   .. literalinclude:: /sdk/boilerplates/test_train_label_regex_tokenizer.py
      :language: python
      :start-after: # start initialize
      :end-before: end initialize
      :dedent: 4

3. **Initialize the ListTokenizer**

   The `ListTokenizer` will hold all the `RegexTokenizers` found for the Label. 

   .. literalinclude:: /sdk/boilerplates/test_train_label_regex_tokenizer.py
      :language: python
      :start-after: start listtokenizer
      :end-before: end listtokenizer
      :dedent: 4

4. **Retrieve the "Lohnart" Label**

   We retrieve the Label using its name. If you have its ID, you could also use the `Project.get_label_by_id` method. 

   .. literalinclude:: /sdk/boilerplates/test_train_label_regex_tokenizer.py
      :language: python
      :start-after: start label
      :end-before: end label
      :dedent: 4

5. **Find Label Regexes and Create RegexTokenizers**

   We then use `Label.find_regex` to algorithmically search for the best fitting regexes matching the Annotations associated with this Label. 

   .. literalinclude:: /sdk/boilerplates/test_train_label_regex_tokenizer.py
      :language: python
      :start-after: start train
      :end-before: end train
      :dedent: 4

6. **Use the new Tokenizer to Create New Annotations**

   Finally, we can use the Tokenizer to create new `NO_LABEL` Annotations which match with the regex patterns found. 

   .. literalinclude:: /sdk/boilerplates/test_train_label_regex_tokenizer.py
      :language: python
      :start-after: start use
      :end-before: end use
      :dedent: 4

.. _paragraph-tokenizer-tutorial:

### Paragraph Tokenizer

The `ParagraphTokenizer` class is a specialized [Tokenizer](https://dev.konfuzio.com/sdk/sourcecode.html#tokenizers) 
designed to create [Annotations](https://dev.konfuzio.com/sdk/sourcecode.html#annotation) splitting a Document into 
meaningful sections. It provides two modes of operation: `detectron` and `line_distance`.

#### Parameters

* `mode`: This parameter determines the mode of operation for the 
[Tokenizer](https://dev.konfuzio.com/sdk/sourcecode.html#tokenizers). It can be `detectron` or `line_distance`. 
In `detectron` mode, the Tokenizer uses a fine-tuned [Detectron2](https://github.com/facebookresearch/detectron2) model 
to assist in Document segmentation. This mode tends to be more accurate but slower as it requires to make an API call to 
the model hosted on Konfuzio servers. The `line_distance` mode, on the other hand, uses a rule-based approach that is 
faster but less accurate, especially with Documents having two columns or other complex layouts. The default is 
`detectron`.

#### Line Distance Approach

The line_distance approach offers a straightforward, efficient way to segment Documents based on line heights, proving 
especially useful for simple, single-column formats. Despite its limitations with complex layouts, its fast processing 
and relatively accurate results make it a practical choice for tasks where speed is a priority and the Document 
structure isn't overly complicated. 

##### Parameters for Line Distance Approach

* `line_height_ratio`: Specifies the ratio of the median line height used as a threshold to create a new paragraph when 
using the Tokenizer in `line_distance` mode. The default value is 0.8. If you notice that the Tokenizer is not creating 
new paragraphs when it should, you can try lowering this value. Alternatively, if the Tokenizer is creating too many 
paragraphs, you can try increasing this value.

* `height`: This optional parameter allows you to define a specific line height threshold for creating new paragraphs. 
If set to None, the Tokenizer uses the intelligently calculated height threshold.

For a quicker result with a relatively simpler, single-column Document, you can use the `ParagraphTokenizer` in 
`line_distance` mode like this:

.. literalinclude:: /sdk/boilerplates/test_paragraph_tokenizer.py
   :language: python
   :start-after: start init2
   :end-before: end init2
   :dedent: 4

.. image:: /_static/img/line_distance_paragraph_tokenizer.png

While the line_distance approach offers efficient Document segmentation for simpler formats, it can struggle with complex 
layouts and diverse Document elements. The computer vision approach, using tools like Detectron2, adapts to a wide range 
of layouts and provides precise segmentation even in complex scenarios. Although it may require more processing power, 
the significant improvement in accuracy and comprehensiveness makes it a powerful upgrade for high-quality Document analysis.

#### Computer Vision Approach

With the computer vision approach, we can create Labels, identify figures, tables, lists, texts, and titles, thereby giving 
us a comprehensive understanding of the Document's structure. 

Using the computer vision approach might require more processing power and might be slower compared to the line_distance 
approach, but the significant leap in the comprehensiveness of the output makes it a powerful tool.

##### Parameters for Computer Vision Approach

* `create_detectron_labels`: This boolean flag determines whether to apply Labels given by the Detectron2 model when 
the Tokenizer is used in `detectron` mode. If the Labels don't exist, they will be created. The potential Labels include 
`figure`, `table`, `list`, `text` and `title`. If this option is set to False, the Tokenizer will create `NO_LABEL` 
Annotations, just like with out other [Tokenizers](https://dev.konfuzio.com/sdk/sourcecode.html#tokenizers). The default 
value is False.

To tokenize a Document into paragraphs using the `ParagraphTokenizer` in `detectron` mode and the 
`create_detectron_labels` option to use the Labels provided by our Detectron model, you can use the following code:

.. literalinclude:: /sdk/boilerplates/test_paragraph_tokenizer.py
   :language: python
   :start-after: start init1
   :end-before: end init1
   :dedent: 4

The resulting Annotations will look like this:

.. image:: /_static/img/paragraph_tokenizer.png

Once you have a grasp on how to implement and utilize the ParagraphTokenizer, you open a new world of possibilities. 
The segmentation of Documents doesn't stop at the paragraph level. On the contrary, it is just the beginning. Let's 
embark on a journey to dive deeper into the Document's structure and semantics with the SentenceTokenizer.

### Sentence Tokenizer

The `SentenceTokenizer` class, akin to the :ref:`ParagraphTokenizer<paragraph-tokenizer-tutorial>`, 
is a specialized Tokenizer designed to split a Document into sentences. It also provides two modes of operation: 
`detectron` and `line_distance`. And just like the `ParagraphTokenizer`, you can customize the behavior of the Tokenizer 
by passing using the `mode`, `line_height_ratio`, `height` and `create_detectron_labels` parameters. The distinguishing 
feature of the `SentenceTokenizer` is that it will split the Document into sentences, not paragraphs. 

#### Example Usage

To use it, you can use the following code:

.. literalinclude:: /sdk/boilerplates/test_sentence_tokenizer.py
   :language: python
   :start-after: start import
   :end-before: end import
   :dedent: 4

The resulting Annotations will look like this:

.. image:: /_static/img/sentence_tokenizer.png


### How to Choose Which Tokenizer to Use?

When it comes to Natural Language Processing (NLP), choosing the correct Tokenizer can make a significant impact on 
your system's performance and accuracy. The Konfuzio SDK offers several tokenization options, each suited to different 
tasks:

1. **WhitespaceTokenizer**: Perfect for basic word-level processing. This Tokenizer breaks text into chunks separated 
by white spaces. It is ideal for straightforward tasks such as basic keyword extraction.

2. **Label-Specific Regex Tokenizer**: Known as "Character" detection mode on the Konfuzio platform, this Tokenizer 
offers more specialized functionality. It uses Annotations of a Label within a training set to pinpoint and tokenize 
precise chunks of text. It's especially effective for tasks like entity recognition, where accuracy is paramount. By 
recognizing specific word or character patterns, it allows for more precise and nuanced data processing.

3. **ParagraphTokenizer**: Identifies and separates larger text chunks - paragraphs. This is beneficial when your 
text's interpretation relies heavily on the context at the paragraph level.

4. **SentenceTokenizer**: Segments text into sentences. This is useful when the meaning of your text depends on the 
context provided at the sentence level.

Choosing the right Tokenizer is a matter of understanding your NLP task, the structure of your data, and the degree of 
detail your processing requires. By aligning these elements with the functionalities provided by the different 
Tokenizers in the Konfuzio SDK, you can select the best tool for your task.

#### Verify that a Tokenizer finds all Labels

To help you choose the right Tokenizer for your task, it can be useful to try out different Tokenizers and see which 
Spans are found by which Tokenizer. The `Label` class provides a method called `spans_not_found_by_tokenizer` that 
can he helpful in this regard.

Here is an example of how to use the `Label.spans_not_found_by_tokenizer` method. This will allow you to determine if a 
RegexTokenizer is suitable at finding the Spans of a Label, or what Spans might have been annotated wrong. Say, you 
have a number of Annotations assigned to the `Austellungsdatum` Label and want to know which Spans would not be found 
when using the Whitespace Tokenizer. You can follow this example to find all the relevant Spans.

.. literalinclude:: /sdk/boilerplates/test_spans_not_found_label.py
   :language: python
   :start-after: start spans
   :end-before: end spans
   :dedent: 4