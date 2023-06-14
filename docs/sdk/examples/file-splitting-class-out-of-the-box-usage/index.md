.. _file-splitting-tutorials:

## File Splitting 

You can train your own File Splitting AI on the data from any Project of your choice. For that purpose, there are 
several tools in the SDK that enable processing Documents that consist of multiple files and propose splitting them 
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

### Train a File Splitting AI locally

Let's see how to use the `konfuzio_sdk` to automatically split a file into several Documents. We will be using 
a pre-built class `SplittingAI` and an instance of a trained `ContextAwareFileSplittingModel`. The latter uses a 
context-aware logic. By context-aware we mean a rule-based approach that looks for common strings between the first 
Pages of all Category's Documents. Upon predicting whether a Page is a potential splitting point (meaning whether it is 
first or not), we compare Page's contents to these common first-page strings; if there is occurrence of at least one 
such string, we mark a Page to be first (thus meaning it is a splitting point).

This tutorial can also be used with the `MultimodalFileSplittingModel`; the only difference in the initialization is 
that it does not require specifying a Tokenizer explicitly. 

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

.. literalinclude:: /sdk/boilerplates/test_file_splitting_example.py
   :language: python
   :start-after: start upload
   :end-before: end upload
   :dedent: 4
