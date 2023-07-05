### Develop and save a Context-Aware File Splitting AI

In this tutorial, you will learn how to train a custom File Splitting AI on the data from a Project of your choice and 
save a trained model for further usage.

#### Intro

It's common for multi-paged files to not be perfectly organized, and in some cases, multiple independent Documents may be 
included in a single file. To ensure that these Documents are properly processed and separated, we will be discussing a 
method for identifying and splitting them into individual, independent Sub-documents.

.. image:: /sdk/examples/file-splitting-class/multi_file_document_example.png

_Multi-file Document Example_

Konfuzio SDK offers two ways for separating Documents that may be included in a single file. One of them is training 
the instance of the Multimodal File Splitting Model for file splitting that would predict whether a Page is first or 
not and running the Splitting AI with it. Multimodal File Splitting Model is a combined approach based on architecture 
that processes textual and visual data from the Documents separately (in our case, using BERT and VGG19 simplified 
architectures respectively) and then combines the outputs which go into a Multi-layer Perceptron architecture as 
inputs. A more detailed scheme of the architecture can be found further.

If you hover over the image you can zoom or use the full page mode.

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers tags lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;url&quot;:&quot;https://raw.githubusercontent.com/konfuzio-ai/konfuzio-sdk/master/docs/sdk/examples/file-splitting-class/fusion_model.drawio&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/embed2.js?&fetch=https%3A%2F%2Fraw.githubusercontent.com%2Fkonfuzio-ai%2Fkonfuzio-sdk%2Fmaster%2Fdocs%2Fsdk%2Fexamples%2Ffile-splitting-class%2Ffusion_model.drawio"></script>

Another approach is context-aware file splitting logic which is presented by Context Aware File Splitting Model. This 
approach involves analyzing the contents of each Page and identifying similarities to the first Pages of the Document. 
It will allow us to define splitting points and divide the Document into multiple Sub-documents. It's important to note 
that this approach is only effective for Documents written in the same language and that the process must be repeated 
for each Category. In this tutorial, we will explain how to implement the class for this model step by step.

If you are unfamiliar with the SDK's main concepts (like Page or Span), you can get to know them at [Data Layer Concepts](https://dev.konfuzio.com/sdk/explanations.html#data-layer-concepts).


#### Quick explanation

The first step in implementing this method is "training": this involves tokenizing the Document by splitting its text 
into parts, specifically into strings without line breaks. We then gather the exclusive strings from Spans, which are 
the parts of the text in the Page, and compare them to the first Pages of each Document in the training data.

Once we have identified these strings, we can use them to determine whether a Page in an input Document is a first Page 
or not. We do this by going through the strings in the Page and comparing them to the set of strings collected in the 
training stage. If we find at least one string that intersects between the current Page and the strings from the first 
step, we believe it is the first Page.

Note that the more Documents we use in the training stage, the less intersecting strings we are likely to find. If you 
find that your set of first-page strings is empty, try using a smaller slice of the dataset instead of the whole set. 
Generally, when used on Documents within the same Category, this algorithm should not return an empty set. If that is 
the case, it's worth checking if your data is consistent, for example, not in different languages or containing other 
Categories.

#### Step-by-step explanation

In this section, we will walk you through the process of setting up the `ContextAwareFileSplittingModel` class, which 
can be found in the code block at the bottom of this page. This class is already implemented and can be imported using 
`from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel`.

Note that any custom File Splitting AI (derived from `AbstractFileSplittingModel` class) requires having the following 
methods implemented:
- `__init__` to initialize key variables required by the custom AI;
- `fit` to define architecture and training that the model undergoes, i.e. a certain NN architecture or a custom 
- hardcoded logic;
- `predict` to define how the model classifies Pages as first or non-first. **NB:** the classification needs to be 
run on the Page level, not the Document level – the result of classification is reflected in `is_first_page` attribute 
value, which is unique to the Page class and is not present in Document class. Pages with `is_first_page = True` become 
splitting points, thus, each new Sub-Document has a Page predicted as first as its starting point.

To begin, we will make all the necessary imports:

.. literalinclude:: /sdk/boilerplates/test_file_splitting_example.py
   :language: python
   :start-after: start imports
   :end-before: end imports
   :dedent: 4

Then, let's initialize the `ContextAwareFileSplittingModel` class:

.. literalinclude:: ../../konfuzio_sdk/trainer/file_splitting.py
   :language: python
   :start-after: begin init
   :end-before: end init
   :dedent: 4


The class inherits from `AbstractFileSplittingModel`, so we run `super().__init__(categories=categories)` to properly 
inherit its attributes. The `tokenizer` attribute will be used to process the text within the Document, separating it 
into Spans. This is done to ensure that the text in all the Documents is split using the same logic (particularly 
tokenization by separating on `\n` whitespaces by ConnectedTextTokenizer, which is used in the example in the end of the 
page) and it will be possible to find common Spans. It will be used for training and testing Documents as well as any 
Document that will undergo splitting. It's important to note that if you run fitting with one Tokenizer and then 
reassign it within the same instance of the model, all previously gathered strings will be deleted and replaced by new 
ones. `requires_images` and `requires_text` determine whether these types of data are used for prediction; this is 
needed for distinguishing between preprocessing types once a model is passed into the Splitting AI.   

An example of how ConnectedTextTokenizer works:

.. exec_code::
   
   # --- hide: start ---
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
   # --- hide: stop ---
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
   # --- hide: start ---
   assert len(test_document.spans()) == 6
   # --- hide: stop ---
   # Start string
   test_document.spans()[0].offset_string

   # output: "Hi all,"
   # --- hide: start ---
   assert test_document.spans()[0].offset_string == 'Hi all,'


The first method to define will be the `fit()` method. For each Category, we call `exclusive_first_page_strings` method, 
which allows us to gather the strings that appear on the first Page of each Document. `allow_empty_categories` allows 
for returning empty lists for Categories that haven't had any exclusive first-page strings found across their Documents. 
This means that those Categories would not be used in the prediction process.

.. literalinclude:: ../../konfuzio_sdk/trainer/file_splitting.py
   :language: python
   :start-after: begin fit
   :end-before: end fit
   :dedent: 4

Next, we define `predict()` method. The method accepts a Page as an input and checks its Span set for containing 
first-page strings for each of the Categories. If there is at least one intersection, the Page is predicted to be a 
first Page. If there are no intersections, the Page is predicted to be a non-first Page.

.. literalinclude:: ../../konfuzio_sdk/trainer/file_splitting.py
   :language: python
   :start-after: begin predict
   :end-before: end predict
   :dedent: 4

Lastly, a `check_is_ready()` method is defined. This method is used to ensure that a model is ready for prediction: the
checks cover that the Tokenizer and a set of Categories is defined, and that at least one of the Categories has 
exclusive first-page strings.

.. literalinclude:: ../../konfuzio_sdk/trainer/file_splitting.py
   :language: python
   :start-after: begin check
   :end-before: end check
   :dedent: 4

Full code of class:

.. literalinclude:: ../../konfuzio_sdk/trainer/file_splitting.py
   :language: python
   :start-after: begin class (this and further comments are for the documentation)
   :end-before: end check

A quick example of the class's usage:

.. literalinclude:: /sdk/boilerplates/test_file_splitting_example.py
   :language: python
   :start-after: start file splitting
   :end-before: end file splitting

