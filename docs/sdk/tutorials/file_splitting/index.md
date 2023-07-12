.. _file-splitting-tutorials:

## File Splitting 

PDFs often encapsulate multiple distinct Documents within a single file, leading to complex navigation and information 
retrieval. Document splitting tackles this by disentangling these intertwined files into separate Documents. This 
guide introduces you to tools and models that automate this process, streamlining your work with multi-Document PDFs.

### Overview

You can train your own File Splitting AI on the data from any Project of your choice ([data preparation tutorial here](tutorials.html#tutorials.html#prepare-the-data-for-training-and-testing-the-ai)). 
Note that Pages in all the Documents used for training and testing have to be ordered correctly – that is to say, not 
mixed up in order. The ground-truth first Page of each Document should go first in the file, ground-truth second Page 
goes second and so on. This is needed because the Splitting AI operates on the idea that the splitting points in a 
stream of Pages are the starting Pages of each Sub-Document in the stream.

For that purpose, there are several tools in the SDK that enable processing Documents that consist of multiple files and propose splitting them 
into the Sub-Documents accordingly:

- A Context Aware File Splitting Model uses a simple hands-on logic based on scanning Category's Documents and finding
strings exclusive for first Pages of all Documents within the Category. Upon predicting whether a Page is a potential
splitting point (meaning whether it is first or not), we compare Page's contents to these exclusive first-page strings;
if there is occurrence of at least one such string, we mark a Page to be first (thus meaning it is a splitting point).
An instance of the Context Aware File Splitting Model can be used to initially build a File Splitting pipeline and can
later be replaced with more complex solutions.

  A Context Aware File Splitting Model instance can be used with an interface provided by Splitting AI – this class
accepts a whole Document instead of a single Page and proposes splitting points or splits the original Documents.


- A Multimodal File Splitting Model is a model that uses an approach that takes both visual and textual parts of the
Pages and processes them independently via the combined VGG19 architecture (simplified) and LegalBERT, passing the
resulting outputs together to a Multi-Layered Perceptron. Model's output is also a prediction of a Page being first or
non-first.

For developing a custom File Splitting approach, we propose an abstract class `AbstractFileSplittingModel`.

### Train a Context Aware File Splitting AI

Let's see how to use the `konfuzio_sdk` to automatically split a file into several Documents. We will be using 
a pre-built class `SplittingAI` and an instance of a trained `ContextAwareFileSplittingModel`. The latter uses a 
context-aware logic. By context-aware we mean a rule-based approach that looks for common strings between the first 
Pages of all Category's Documents. Upon predicting whether a Page is a potential splitting point (meaning whether it is 
first or not), we compare Page's contents to these common first-page strings; if there is occurrence of at least one 
such string, we mark a Page to be first (thus meaning it is a splitting point).

.. literalinclude:: /sdk/boilerplates/test_file_splitting_example.py
   :language: python
   :start-after: start imports
   :end-before: end imports
   :dedent: 4
.. literalinclude:: /sdk/boilerplates/test_file_splitting_example.py
   :language: python
   :start-after: start file splitting
   :end-before: end file splitting
   :dedent: 4

After you have trained your custom AI, you can upload it using the steps from the [tutorial](https://help.konfuzio.com/tutorials/migrate-trained-ai-to-an-new-project-to-annotate-documents-faster/index.html#upload-extraction-or-category-ai-to-target-instance)
or using the method `upload_ai_model()`.

For the first option, go to the Superuser AIs and select your locally stored pickle file, setting Model Type to 
Splitting and status to Training finished, then save the AI. After that, go to the Splitting AIs, choose your AI and 
select an action "Activate Splitting AI".

For the second option, provide the path to your model to the `upload_ai_model()`. You can also remove an uploaded model
by using `delete_ai_model()`.

```python
 from konfuzio_sdk.api import upload_ai_model, delete_ai_model

 # upload a saved model to the server
 model_id = upload_ai_model(save_path)

 # remove model
 delete_ai_model(model_id, ai_type='file_splitting')
```

### Train a Multimodal File Splitting AI

The above tutorial for the `ContextAwareFileSplittingModel` can also be used with the `MultimodalFileSplittingModel`. 
The only difference is that the `MultimodalFileSplittingModel` does not need to be initialized with a Tokenizer.

### Develop and save a Context-Aware File Splitting AI

If the solutions presented above do not meet your requirements, we also allow the training of custom File Splitting AIs 
on the data from a Project of your choice. You can then save your trained model and use it with Konfuzio.

#### Intro

It's common for multi-paged files to not be perfectly organized, and in some cases, multiple independent Documents may be 
included in a single file. To ensure that these Documents are properly processed and separated, we will be discussing a 
method for identifying and splitting them into individual, independent Sub-documents.

.. image:: /sdk/tutorials/file_splitting/multi_file_document_example.png

_Multi-file Document Example_

Konfuzio SDK offers two ways for separating Documents that may be included in a single file. One of them is training 
the instance of the Multimodal File Splitting Model for file splitting that would predict whether a Page is first or 
not and running the Splitting AI with it. Multimodal File Splitting Model is a combined approach based on architecture 
that processes textual and visual data from the Documents separately (in our case, using BERT and VGG19 simplified 
architectures respectively) and then combines the outputs which go into a Multi-layer Perceptron architecture as 
inputs. A more detailed scheme of the architecture can be found further.

If you hover over the image you can zoom or use the full page mode.

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers tags lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;url&quot;:&quot;https://raw.githubusercontent.com/konfuzio-ai/konfuzio-sdk/master/docs/sdk/tutorials/file_splitting/fusion_model.drawio&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/embed2.js?&fetch=https%3A%2F%2Fraw.githubusercontent.com%2Fkonfuzio-ai%2Fkonfuzio-sdk%2Fmaster%2Fdocs%2Fsdk%2Ftutorials%2Ffile_splitting%2Ffusion_model.drawio"></script>

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

.. literalinclude:: /sdk/boilerplates/test_connected_text_tokenizer.py
   :language: python
   :start-after: Start tokenize
   :end-before: End tokenize
   :dedent: 4
.. literalinclude:: /sdk/boilerplates/test_connected_text_tokenizer.py
   :language: python
   :start-after: Start string
   :end-before: End string
   :dedent: 4

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
   :pyobject: ContextAwareFileSplittingModel

A quick example of the class's usage:

.. literalinclude:: /sdk/boilerplates/test_file_splitting_example.py
   :language: python
   :start-after: start file splitting
   :end-before: end file splitting
   :dedent: 4

### Create a custom File Splitting AI

This section explains how to train a custom File Splitting AI locally, how to save it and upload it to the Konfuzio 
Server. If you run this tutorial in Colab and experience any version compatibility issues when working with the SDK, restart the
runtime and initialize the SDK once again; this will resolve the issue.

Note: you don't necessarily need to create the AI from scratch if you already have some document-processing architecture.
You just need to wrap it into the class that corresponds to our File Splitting AI structure. Follow the steps in this 
tutorial to find out what are the requirements for that.

By default, any [File Splitting AI](sourcecode.html#file-splitting-ai) class should derive from the 
`AbstractFileSplittingModel` class and implement the following methods:

.. literalinclude:: /sdk/boilerplates/test_custom_file_splitting_ai.py
   :language: python
   :start-after: start class
   :end-before: end class
   :dedent: 4

### Evaluate a File Splitting AI

`FileSplittingEvaluation` class can be used to evaluate performance of Context-Aware File Splitting Model, returning a 
set of metrics that includes precision, recall, f1 measure, True Positives, False Positives, True Negatives, and False 
Negatives. 

The class's methods `calculate()` and `calculate_by_category()` are run at initialization. The class receives two lists 
of Documents as an input – first list consists of ground-truth Documents where all first Pages are marked as such, 
second is of Documents on Pages of which File Splitting Model ran a prediction of them being first or non-first. 

The initialization would look like this:

.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start eval_example
   :end-before: end eval_example
   :dedent: 4

The class compares each pair of Pages. If a Page is labeled as first and the model also predicted it as first, it is 
considered a True Positive. If a Page is labeled as first but the model predicted it as non-first, it is considered a 
False Negative. If a Page is labeled as non-first but the model predicted it as first, it is considered a False 
Positive. If a Page is labeled as non-first and the model also predicted it as non-first, it is considered a True 
Negative.

|  | predicted correctly | predicted incorrectly |
| ------ | ------ | ------ |
|    first Page    |    TP    | FN |
|    non-first Page    |   TN     | FP |

After iterating through all Pages of all Documents, precision, recall and f1 measure are calculated. If you wish to set 
metrics to `None` in case there has been an attempt of zero division, set `allow_zero=True` at the initialization.

To see a certain metric after the class has been initialized, you can call a metric's method:

.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start single_metric
   :end-before: end single_metric
   :dedent: 4

It is also possible to look at the metrics calculated by each Category independently. For this, pass 
`search=YOUR_CATEGORY_HERE` when calling the wanted metric's method: 

.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start metric_category
   :end-before: end metric_category
   :dedent: 4

For more details, see the [Python API Documentation](https://dev.konfuzio.com/sdk/sourcecode.html#ai-evaluation) on 
Evaluation.

#### Example of evaluation input and output 

Suppose in our test dataset we have 2 Documents of 2 Categories: one 3-paged, consisting of a single file (-> it has 
only one ground-truth first Page) of a first Category, and one 5-paged, consisting of three files: two 2-paged and one 
1-paged (-> it has three ground-truth first Pages), of a second Category.

.. image:: /sdk/tutorials/file_splitting/document_example_1.png

_First document_

.. image:: /sdk/tutorials/file_splitting/document_example_2.png

_Second document_

.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start document creation
   :end-before: end document creation
   :dedent: 4

We need to pass two lists of Documents into the `FileSplittingEvaluation` class. So, before that, we need to run each 
Page of the Documents through the model's prediction.

Let's say the evaluation gave good results, with only one first Page being predicted as non-first and all the other 
Pages being predicted correctly. An example of how the evaluation would be implemented would be:

.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start splitting
   :end-before: end splitting
   :dedent: 4
.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start scores
   :end-before: end scores
   :dedent: 4

Our results could be reflected in a following table:

| TPs | TNs | FPs | FNs | precision | recall | F1    |
| ---- |-----|-----| ----- | ---- | ---- |-------|
| 3 | 4   |  0  | 1 | 1 | 0.75 | 0.85  |

If we want to see evaluation results by Category, the implementation of the Evaluation would look like this:

.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start scores_category
   :end-before: end scores_category
   :dedent: 4

the output could be reflected in a following table:

| Category | TPs | TNs | FPs | FNs | precision | recall | F1  |
| ---- |-----|-----|-----|-----| ---- |--------|-----|
| Category 1 | 1   | 2   | 0   | 0   | 1 | 1      | 1   |
| Category 2 | 2   | 2   | 0   | 1   | 1 | 0.66   | 0.8 |

To log metrics after evaluation, you can call `EvaluationCalculator`'s method `metrics_logging` (you would need to 
specify the metrics accordingly at the class's initialization). Example usage:

.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start calculator
   :end-before: end calculator
   :dedent: 4