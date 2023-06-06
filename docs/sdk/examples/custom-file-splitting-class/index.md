### Create a custom File Splitting AI

This section explains how to train a custom File Splitting AI locally, how to save it and upload it to the Konfuzio 
Server. 

By default, any [File Splitting AI](sourcecode.html#file-splitting-ai) class should derive from the 
`AbstractFileSplittingModel` class and implement the following interface:

.. literalinclude:: /sdk/boilerplates/test_custom_file_splitting_ai.py
   :language: python
   :start-after: start class
   :end-before: end class
   :dedent: 4

