## File Splitting

### Split a file into separate Documents

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
   :lines: 6,8-11,18-72
